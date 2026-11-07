#![no_main]
use libfuzzer_sys::fuzz_target;

fuzz_target!(|data: &[u8]| {
    dfin_fuzz::tunnel_packet(data);
});
