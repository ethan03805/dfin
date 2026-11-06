//! `dfin`: one binary, several roles.
//!
//! * `dfin stage`  serves a contiguous layer range of a GGUF to a client daemon.
//! * `dfin client` runs the head and tail locally, drives a route of stages, and
//!   exposes an OpenAI-compatible API on localhost.
//! * `dfin relay`  pipes streams between a client and its route so stages never
//!   learn the client's address.
//! * `dfin hub`    publishes a signed directory of announced stages.
//! * `dfin fetch`  downloads the shards a piece needs from nodes or seeds.
//! * `dfin pod`    launches stages plus a client on this machine for testing.
//! * `dfin bench`  measures monolithic versus pipelined decode on this machine.

mod bench;
mod client;
mod config;
mod fetch;
mod hub;
mod pod;
mod relay;
mod shards;
mod stage;

use clap::{Parser, Subcommand};

#[derive(Parser, Debug)]
#[command(
    name = "dfin",
    version,
    about = "Decentralized pipeline-parallel inference node"
)]
struct Cli {
    /// Log filter, e.g. `info`, `debug`, `dfin=debug,llama=warn`.
    #[arg(long, env = "DFIN_LOG", default_value = "info", global = true)]
    log: String,

    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Subcommand, Debug)]
enum Cmd {
    /// Serve one layer range of a model to a client daemon.
    Stage(stage::StageArgs),
    /// Run the client daemon: local head and tail, pipeline routing, OpenAI API.
    Client(client::ClientArgs),
    /// Pipe streams between clients and their routes, hiding client addresses from stages.
    Relay(relay::RelayArgs),
    /// Publish a signed directory of the stages that announce to this hub.
    Hub(hub::HubArgs),
    /// Download the shards of a model that a layer range needs, from nodes or seeds.
    Fetch(fetch::FetchArgs),
    /// Launch a local pod (stages plus client) on this machine.
    Pod(pod::PodArgs),
    /// Benchmark monolithic versus pipelined decode on this machine.
    Bench(bench::BenchArgs),
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let cli = Cli::parse();
    init_tracing(&cli.log);
    match cli.cmd {
        Cmd::Stage(args) => stage::run(args).await,
        Cmd::Client(args) => client::run(args).await,
        Cmd::Relay(args) => relay::run(args).await,
        Cmd::Hub(args) => hub::run(args).await,
        Cmd::Fetch(args) => fetch::run(args).await,
        Cmd::Pod(args) => pod::run(args).await,
        Cmd::Bench(args) => bench::run(args).await,
    }
}

fn init_tracing(filter: &str) {
    use tracing_subscriber::{fmt, EnvFilter};
    let filter = EnvFilter::try_new(filter).unwrap_or_else(|_| EnvFilter::new("info"));
    let _ = fmt()
        .with_env_filter(filter)
        .with_target(true)
        .with_writer(std::io::stderr)
        .try_init();
}
