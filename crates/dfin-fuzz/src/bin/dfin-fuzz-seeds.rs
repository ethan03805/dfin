//! Write each target's seed corpus: `dfin-fuzz-seeds <corpus dir>` makes
//! `<corpus dir>/<target>/seed-<n>` for every target, which is where cargo-fuzz looks.

use std::path::PathBuf;

fn main() -> std::io::Result<()> {
    let dir = std::env::args().nth(1).map(PathBuf::from).unwrap_or_else(|| {
        eprintln!("usage: dfin-fuzz-seeds <corpus dir>");
        std::process::exit(2);
    });
    for (name, target) in dfin_fuzz::TARGETS {
        let sub = dir.join(name);
        std::fs::create_dir_all(&sub)?;
        let seeds = dfin_fuzz::seeds::for_target(name);
        for (i, seed) in seeds.iter().enumerate() {
            assert!(target(seed), "seed {i} of {name} does not decode");
            std::fs::write(sub.join(format!("seed-{i}")), seed)?;
        }
        println!("{name}: {} seeds", seeds.len());
    }
    Ok(())
}
