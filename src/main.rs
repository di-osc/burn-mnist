#![recursion_limit = "131"]
mod data;
mod inference;
mod model;
mod training;

use std::num::NonZeroUsize;

use burn::backend::Autodiff;
use burn::backend::wgpu::{Wgpu, WgpuDevice};
use burn::data::dataset::transform::Window;
use burn::optim::AdamConfig;
use clap::{Parser, Subcommand};

use crate::model::ModelConfig;
use crate::training::TrainingConfig;

#[derive(Parser)]
#[command(version, about, long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// Train the MNIST model.
    Train {
        /// Path to the directory where the model artifacts will be saved.
        #[arg(short, long, default_value = "mnist_artifacts")]
        artifact_dir: String,
        /// Training configuration.
        #[arg(default_value = "10")]
        num_epochs: usize,
        /// Batch size for training.
        #[arg(short, long, default_value = "64")]
        batch_size: usize,
        /// Number of workers for data loading.
        #[arg(short, long, default_value = "4")]
        num_workers: usize,
        /// Random seed for reproducibility.
        #[arg(short, long, default_value = "42")]
        seed: u64,
        /// Learning rate for the optimizer.
        #[arg(long, default_value = "1.0e-4")]
        lr: f64,
    },
    /// Run inference on a sample image.
    Test {
        /// Path to the directory where the model artifacts are saved.
        #[arg(short, long, default_value = "mnist_artifacts")]
        artifact_dir: String,
        /// number of images to infer.
        #[arg(short, long, default_value = "10")]
        num_images: NonZeroUsize,
    },
}

/// Main function to run the training and inference.
///
/// This function initializes the WGPU device, trains the model, and then performs inference on a sample image.
fn main() {
    let cli = Cli::parse();
    match cli.command {
        Commands::Train {
            artifact_dir,
            num_epochs,
            batch_size,
            num_workers,
            seed,
            lr,
        } => {
            type MyBackend = Wgpu<f32, i32>;
            type MyAotudiffBackend = Autodiff<MyBackend>;

            let device = WgpuDevice::DefaultDevice;

            let start = std::time::Instant::now();
            let config = TrainingConfig {
                model: ModelConfig::new(10, 512),
                optimizer: AdamConfig::new(),
                num_epochs: num_epochs,
                batch_size: batch_size,
                num_workers: num_workers,
                seed: seed,
                learning_rate: lr,
            };
            training::train::<MyAotudiffBackend>(&artifact_dir, config, device.clone());
            let duration = start.elapsed();
            println!("Training time: {:?}", duration);
        }
        Commands::Test {
            artifact_dir,
            num_images,
        } => {
            type MyBackend = Wgpu<f32, i32>;
            let device = WgpuDevice::DefaultDevice;
            let num_images = num_images.get();
            let random_start = rand::random::<u8>().into();
            let items = burn::data::dataset::vision::MnistDataset::test()
                .window(random_start, NonZeroUsize::new(num_images).unwrap())
                .unwrap();
            let start = std::time::Instant::now();
            crate::inference::infer::<MyBackend>(&artifact_dir, device, items);
            let duration = start.elapsed();
            println!("Predict {} images in: {:?}", num_images, duration);
        }
    }
}
