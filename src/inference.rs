use crate::data::MnistBatcher;
use crate::training::TrainingConfig;
use burn::data::dataloader::batcher::Batcher;
use burn::record::Recorder;
use burn::{data::dataset::vision::MnistItem, prelude::*, record::CompactRecorder};

pub fn infer<B: Backend>(artifact_dir: &str, device: B::Device, items: Vec<MnistItem>) {
    let config = TrainingConfig::load(format!("{artifact_dir}/config.json"))
        .expect("Config should exist for the model; run train first");
    let record = CompactRecorder::new();
    let record = record
        .load(format!("{artifact_dir}/model").into(), &device)
        .expect("Trained model should exist; run train first");

    let model = config.model.init::<B>(&device).load_record(record);

    let labels = items.iter().map(|item| item.label).collect::<Vec<_>>();
    let batcher = MnistBatcher::default();
    let batch = batcher.batch(items, &device);
    let output = model.forward(batch.images);
    let predicted: Vec<i32> = output
        .argmax(1)
        .flatten::<1>(0, 1)
        .to_data()
        .to_vec()
        .unwrap();
    // to u8 for printing
    let predicted: Vec<u8> = predicted.iter().map(|&x| x as u8).collect();

    let correct = predicted
        .iter()
        .zip(labels.iter())
        .filter(|(pred, label)| *pred == *label)
        .count();
    let accuracy = correct as f64 / labels.len() as f64;
    println!("Accuracy: {:.2}%", accuracy * 100.0);
}
