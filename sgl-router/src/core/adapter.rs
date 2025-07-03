use std::sync::Arc;
use crate::core::worker::{Worker, WorkerFactory};

/// 把 URL 字符串包装成 Regular Worker
pub fn url_to_worker(url: String) -> Arc<dyn Worker> {
    WorkerFactory::create_regular(url)
}

/// 从 Worker 再取出原始 URL（给旧接口用）
pub fn worker_to_url(worker: &Arc<dyn Worker>) -> String {
    worker.url().to_string()
}

/// Vec<String>  → Vec<Arc<dyn Worker>>
pub fn urls_to_workers(urls: &[String]) -> Vec<Arc<dyn Worker>> {
    urls.iter().map(|url| WorkerFactory::create_regular(url.clone())).collect()
}

/// Vec<Arc<dyn Worker>> → Vec<String>
pub fn workers_to_urls(workers: &[Arc<dyn Worker>]) -> Vec<String> {
    workers.iter().map(|w| w.url().to_string()).collect()
}
