use clap::Parser;

mod audio;
mod cli;
mod plot;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    cli::run(cli::Args::parse())
}
