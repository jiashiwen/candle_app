use std::sync::Arc;

use candle_core::{
    utils::{cuda_is_available, metal_is_available},
    Device, Result,
};
use once_cell::sync::Lazy;
use tokio::runtime::{Builder, Runtime};

pub static GLOBAL_RUNTIME: Lazy<Arc<Runtime>> = Lazy::new(|| {
    let runtime = match init_task_runtime() {
        Ok(db) => db,
        Err(err) => panic!("{}", err),
    };
    Arc::new(runtime)
});

fn init_task_runtime() -> Result<Runtime> {
    let rt = Builder::new_multi_thread()
        .worker_threads(num_cpus::get())
        .enable_all()
        .max_io_events_per_tick(32)
        .build()?;
    Ok(rt)
}

pub fn device(cpu: bool) -> Result<Device> {
    if cpu {
        Ok(Device::Cpu)
    } else if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    } else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    } else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!(
                "Running on CPU, to run on GPU(metal), build this example with `--features metal`"
            );
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}

pub async fn hub_load_safetensors(
    repo: &hf_hub::api::tokio::ApiRepo,
    json_file: &str,
) -> Result<Vec<std::path::PathBuf>> {
    let json_file = repo
        .get(json_file)
        .await
        .map_err(candle_core::Error::wrap)?;
    let json_file = std::fs::File::open(json_file)?;
    let json: serde_json::Value =
        serde_json::from_reader(&json_file).map_err(candle_core::Error::wrap)?;
    let weight_map = match json.get("weight_map") {
        None => candle_core::bail!("no weight map in {json_file:?}"),
        Some(serde_json::Value::Object(map)) => map,
        Some(_) => candle_core::bail!("weight map in {json_file:?} is not a map"),
    };
    let mut safetensors_files = std::collections::HashSet::new();
    for value in weight_map.values() {
        if let Some(file) = value.as_str() {
            safetensors_files.insert(file.to_string());
        }
    }
    // let safetensors_files = safetensors_files
    //     .iter()
    //     // .map(|v| repo.get(v).map_err(candle::Error::wrap))
    //     .map(|v| repo.get(v).map_err(candle_core::Error::wrap))
    //     .collect::<Result<Vec<_>>>()?;

    let mut vec_files = vec![];
    for v in safetensors_files.iter() {
        let path_buf = repo.get(v).await.map_err(candle_core::Error::wrap)?;
        vec_files.push(path_buf);
    }
    Ok(vec_files)
    // Ok(safetensors_files)
}
