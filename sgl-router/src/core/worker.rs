// src/core/worker.rs
use async_trait::async_trait;
use std::fmt;
use std::sync::{Arc, RwLock};
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use reqwest::Client;
use std::time::{Duration, Instant};
use std::fmt::Debug;



#[derive(Debug, Clone, PartialEq)]
pub enum WorkerType {
    Regular,
    Prefill { bootstrap_port: Option<u16> },
    Decode,
}

pub enum WorkerError {
    HealthCheckFailed(String),
}

pub const HEALTH_ENDPOINT: &str = "/health";
const HEALTH_TIMEOUT: Duration = Duration::from_secs(2);
const HEALTH_CACHE_TTL: Duration = Duration::from_secs(10);

#[async_trait]
pub trait Worker: Send + Sync + Debug  {
    fn url(&self) -> &str;
    fn worker_type(&self) -> WorkerType;
    fn is_healthy(&self) -> bool;
    async fn check_health(&self) -> Result<(), WorkerError>;
    fn load(&self) -> Arc<AtomicUsize>;
    fn update_health(&self, healthy: bool);
}

#[derive(Clone, Debug)]
pub struct WorkerImpl {
    url: String,
    worker_type: WorkerType,
    healthy: Arc<AtomicBool>,
    load: Arc<AtomicUsize>,
    last_health_check: Arc<RwLock<Instant>>,
}

impl WorkerImpl {
    pub fn new(url: String, worker_type: WorkerType) -> Self {
        Self {
            url,
            worker_type,
            healthy: Arc::new(AtomicBool::new(true)),
            load: Arc::new(AtomicUsize::new(0)),
            last_health_check: Arc::new(RwLock::new(Instant::now())),
        }
    }
}

#[async_trait]
impl Worker for WorkerImpl {
    fn url(&self) -> &str {
        &self.url
    }

    fn worker_type(&self) -> WorkerType {
        self.worker_type.clone()
    }

    fn is_healthy(&self) -> bool {
        self.healthy.load(Ordering::Acquire)
    }

    async fn check_health(&self) -> Result<(), WorkerError> {
        {
            let last = *self.last_health_check.read().unwrap();
            if last.elapsed() < HEALTH_CACHE_TTL {
                // 用已有结果直接返回
                return if self.is_healthy() {
                    Ok(())
                } else {
                    Err(WorkerError::HealthCheckFailed("Cached unhealthy".into()))
                };
            }
        }


        let client = Client::builder()
            .timeout(HEALTH_TIMEOUT)
            .build()
            .map_err(|e| {
                self.update_health(false);
                WorkerError::HealthCheckFailed(format!("Failed to create client: {}", e))
            })?;

        let full_url = format!("{}{}", self.url, HEALTH_ENDPOINT);
        match client.get(&full_url).send().await {
            Ok(res) if res.status().is_success() => {
                self.update_health(true);
                *self.last_health_check.write().unwrap() = Instant::now();
                Ok(())
            }
            _ => {
                self.update_health(false);
                *self.last_health_check.write().unwrap() = Instant::now();
                Err(WorkerError::HealthCheckFailed("…".into()))
            }
        }
    }

    fn load(&self) -> Arc<AtomicUsize> {
        Arc::clone(&self.load)
    }

    fn update_health(&self, healthy: bool) {
        self.healthy.store(healthy, Ordering::Release);
    }
}

pub struct WorkerFactory;

impl WorkerFactory {
    pub fn create_regular(url: String) -> Arc<dyn Worker> {
        Arc::new(WorkerImpl::new(url, WorkerType::Regular))
    }

    pub fn create_prefill(url: String, bootstrap_port: Option<u16>) -> Arc<dyn Worker> {
        Arc::new(WorkerImpl::new(url, WorkerType::Prefill { bootstrap_port }))
    }

    pub fn create_decode(url: String) -> Arc<dyn Worker> {
        Arc::new(WorkerImpl::new(url, WorkerType::Decode))
    }
}
