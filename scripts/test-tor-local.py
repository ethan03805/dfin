#!/usr/bin/env python3
"""Run dfin's opt-in integration test through a private loopback Tor network.

Requires verified tor and tor-gencert binaries, Cargo, the built inference backend,
and models/tiny-llama-f32.gguf. No public Tor relays, exits, or onion services are used.
Logs/configs are retained. All children started here are stopped on exit.

Testing parameters follow Tor's Chutney hs-v3-min topology and torrc generator:
https://gitlab.torproject.org/tpo/core/chutney (dbc640bf4863d4e3d6cb1ad9879b41f9f5862c4e).
These weak, fast-bootstrap settings are ONLY for a private test, never deployment.
"""
import argparse
import json
import os
from pathlib import Path
import re
import socket
import subprocess
import tempfile
import time

NO_WINDOW = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0


def quote(path):
    return json.dumps(str(Path(path).resolve()))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tor", required=True, type=Path)
    parser.add_argument("--tor-gencert", type=Path)
    parser.add_argument("--work-dir", type=Path, help="New directory for configs and logs (must not exist)")
    parser.add_argument("--timeout", type=int, default=240, help="Bootstrap deadline in seconds")
    args = parser.parse_args()
    tor = args.tor.resolve()
    gencert = (args.tor_gencert or tor.with_name("tor-gencert" + tor.suffix)).resolve()
    repo = Path(__file__).resolve().parents[1]
    if not (repo / "models/tiny-llama-f32.gguf").is_file():
        raise SystemExit("Generate models/tiny-llama-f32.gguf before running this test")
    if args.work_dir:
        root = args.work_dir.resolve()
        root.mkdir(parents=True, exist_ok=False)
    else:
        root = Path(tempfile.mkdtemp(prefix="dfin-private-tor-"))
    print(f"Private Tor test files: {root}", flush=True)
    children, logs, reserved = [], [], []

    def port():
        sock = socket.socket()
        sock.bind(("127.0.0.1", 0))
        reserved.append(sock)
        return sock.getsockname()[1]

    def run(command, **kwargs):
        result = subprocess.run([str(x) for x in command], text=True, capture_output=True,
                                creationflags=NO_WINDOW, timeout=60, **kwargs)
        if result.returncode:
            raise RuntimeError(f"Command failed: {command[0]}\n{result.stdout}\n{result.stderr}")
        return result.stdout

    try:
        version = run([tor, "--version"])
        (root / "tor-version.txt").write_text(version)
        print(version.splitlines()[0], flush=True)
        nodes = []
        authorities = []
        for i in range(9):  # Four authorities, three non-exit relays, client, services.
            data = root / f"node{i}"
            (data / "keys").mkdir(parents=True)
            node = {"dir": data, "or": port(), "directory": port(), "socks": port()}
            nodes.append(node)
            if i < 7:
                run([tor, "--ignore-missing-torrc", "-f", data / "not-yet-created",
                     "--orport", "1", "--datadirectory", data, "--nickname", f"test{i}", "--list-fingerprint"])
                node["fingerprint"] = (data / "fingerprint").read_text().split()[1]
            if i < 4:
                run([gencert, "--create-identity-key", "--passphrase-fd", "0",
                     "-i", data / "keys/authority_identity_key", "-s", data / "keys/authority_signing_key",
                     "-c", data / "keys/authority_certificate", "-m", "1", "-a", f"127.0.0.1:{node['directory']}"], input="\n")
                certificate = (data / "keys/authority_certificate").read_text()
                identity = re.search(r"^fingerprint ([A-F0-9]{40})$", certificate, re.MULTILINE).group(1)
                authorities.append(f"DirAuthority test{i} orport={node['or']} no-v2 v3ident={identity} 127.0.0.1:{node['directory']} {node['fingerprint']}")
        services = {name: {"local": f"127.0.0.1:{port()}"} for name in ["hub", "s1", "s2", "relay"]}
        for i, node in enumerate(nodes):
            data = node["dir"]
            lines = ["TestingTorNetwork 1", f"DataDirectory {quote(data)}", f"Nickname test{i}",
                     "RunAsDaemon 0", "Address 127.0.0.1", "AddressDisableIPv6 1", "UseDefaultFallbackDirs 0",
                     "ControlPort 0", "DNSPort 0", "SafeLogging 1", "ExitRelay 0", "ExitPolicy reject *:*",
                     "ClientRejectInternalAddresses 0", "UseMicrodescriptors 0", "ShutdownWaitLength 0",
                     "Log notice stdout", *authorities]
            lines += [f"SocksPort 127.0.0.1:{node['socks']} IsolateSOCKSAuth"] if i >= 7 else ["SocksPort 0"]
            if i < 7:
                lines += [f"ORPort 127.0.0.1:{node['or']}", f"DirPort 127.0.0.1:{node['directory']}"]
            else:
                lines += ["ORPort 0", "DirPort 0", "ClientOnly 1"]
            if i < 4:
                lines += ["AuthoritativeDirectory 1", "V3AuthoritativeDirectory 1", "ContactInfo local-test",
                          "AssumeReachable 1", "TestingV3AuthInitialVotingInterval 20",
                          "TestingV3AuthInitialVoteDelay 4", "TestingV3AuthInitialDistDelay 4",
                          "V3AuthVotingInterval 20", "V3AuthVoteDelay 4", "V3AuthDistDelay 4",
                          "TestingDirAuthVoteGuard *", "TestingDirAuthVoteHSDir *", "V3AuthNIntervalsValid 2",
                          "AuthDirVoteStableGuaranteeMinUptime 0", "ConsensusParams cc_alg=2"]
            if i == 8:
                for name, service in services.items():
                    lines += [f"HiddenServiceDir {quote(data / name)}", "HiddenServiceVersion 3",
                              f"HiddenServicePort 7001 {service['local']}"]
            config = data / "torrc"
            config.write_text("\n".join(lines) + "\n")
            node["config"] = config
            run([tor, "-f", config, "--verify-config"])
        for sock in reserved:
            sock.close()
        reserved.clear()
        for node in nodes:
            log = (node["dir"] / "stdout.log").open("w")
            logs.append(log)
            children.append(subprocess.Popen([str(tor), "-f", str(node["config"])],
                            stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, creationflags=NO_WINDOW))
        print("Started 9 Tor nodes on loopback; waiting for private consensus and circuits.", flush=True)
        deadline = time.monotonic() + args.timeout
        while True:
            if any(p.poll() is not None for p in children):
                raise RuntimeError(f"A Tor process exited; inspect logs in {root}")
            ready = all((n["dir"] / "stdout.log").exists() and
                        "Bootstrapped 100%" in (n["dir"] / "stdout.log").read_text(errors="replace")
                        for n in nodes[7:])
            if ready and all((nodes[8]["dir"] / name / "hostname").is_file() for name in services):
                break
            if time.monotonic() >= deadline:
                raise TimeoutError(f"Private Tor bootstrap timed out; inspect {root}")
            time.sleep(1)
        for name, service in services.items():
            service["onion"] = (nodes[8]["dir"] / name / "hostname").read_text().strip() + ":7001"
        config = root / "dfin-test.json"
        config.write_text(json.dumps({"proxy": f"127.0.0.1:{nodes[7]['socks']}", **services}, indent=2))
        env = dict(os.environ, DFIN_TOR_TEST_CONFIG=str(config))
        print("Private Tor bootstrapped. Running discovery and inference in all return modes.", flush=True)
        command = ["cargo", "test", "--release", "--locked", "-p", "dfin", "--test", "anonymity",
                   "real_tor_discovery_and_inference", "--", "--ignored", "--nocapture"]
        with (root / "dfin-test.log").open("w") as output:
            child = subprocess.Popen(command, cwd=repo, env=env, stdout=output, stderr=subprocess.STDOUT, creationflags=NO_WINDOW)
            children.append(child)
            code = child.wait(timeout=600)
        print((root / "dfin-test.log").read_text(errors="replace"), flush=True)
        if code:
            raise RuntimeError(f"dfin test failed ({code}); logs: {root}")
        print("PASS: real Tor private-network discovery and inference; all test children will stop.", flush=True)
    finally:
        for child in reversed(children):
            if child.poll() is None:
                child.terminate()
        for child in reversed(children):
            try:
                child.wait(timeout=5)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait(timeout=5)
        for log in logs:
            log.close()
        for sock in reserved:
            sock.close()


if __name__ == "__main__":
    main()
