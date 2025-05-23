#![recursion_limit = "131"]
mod data;
mod inference;
mod model;
mod training;

use crate::model::ModelConfig;
use crate::training::TrainingConfig;

use burn::backend::{Autodiff, Wgpu};
use burn::data::dataset::Dataset;
use burn::optim::AdamConfig;

fn main() {
    type MyBackend = Wgpu<f32, i32>;
    type MyAotudiffBackend = Autodiff<MyBackend>;

    let device = burn::backend::wgpu::WgpuDevice::default();

    let artifact_dir = "checkpoints";
    let start = std::time::Instant::now();
    training::train::<MyAotudiffBackend>(
        artifact_dir,
        TrainingConfig::new(ModelConfig::new(10, 512), AdamConfig::new()),
        device.clone(),
    );
    let duration = start.elapsed();
    println!("Training time: {:?}", duration);
    crate::inference::infer::<MyBackend>(
        artifact_dir,
        device,
        burn::data::dataset::vision::MnistDataset::test()
            .get(40)
            .unwrap(),
    );
}
