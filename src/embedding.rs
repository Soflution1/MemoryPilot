/// MemoryPilot v4.0 — Dual Embedding Engine.
/// Primary: fastembed transformer (all-MiniLM-L6-v2, 384-dim) when feature enabled.
/// Fallback: TF-IDF hashing (384-dim) — pure Rust, zero external model.
use std::collections::HashMap;

const VECTOR_DIM: usize = 384;

#[cfg(feature = "fastembed")]
use std::sync::{Mutex, OnceLock};

#[cfg(feature = "fastembed")]
static FASTEMBED_MODEL: OnceLock<Option<Mutex<fastembed::TextEmbedding>>> = OnceLock::new();

#[cfg(feature = "fastembed")]
fn get_fastembed() -> Option<&'static Mutex<fastembed::TextEmbedding>> {
    FASTEMBED_MODEL.get_or_init(|| {
        let opts = fastembed::InitOptions::new(fastembed::EmbeddingModel::AllMiniLML6V2)
            .with_show_download_progress(false);
        match fastembed::TextEmbedding::try_new(opts) {
            Ok(model) => Some(Mutex::new(model)),
            Err(e) => {
                eprintln!("[MemoryPilot] fastembed init failed ({}), using TF-IDF fallback", e);
                None
            }
        }
    }).as_ref()
}

pub fn embed_text(text: &str) -> Vec<f32> {
    #[cfg(feature = "fastembed")]
    {
        if let Some(mtx) = get_fastembed() {
            if let Ok(mut model) = mtx.lock() {
                if let Ok(mut embeddings) = model.embed(vec![text], None) {
                    if let Some(emb) = embeddings.pop() {
                        if emb.len() == VECTOR_DIM {
                            return emb;
                        }
                    }
                }
            }
        }
    }
    tfidf_embed(text)
}

pub fn embed_batch(texts: &[&str]) -> Vec<Vec<f32>> {
    #[cfg(feature = "fastembed")]
    {
        if let Some(mtx) = get_fastembed() {
            if let Ok(mut model) = mtx.lock() {
                if let Ok(embeddings) = model.embed(texts.to_vec(), None) {
                    if embeddings.len() == texts.len() && embeddings.iter().all(|e| e.len() == VECTOR_DIM) {
                        return embeddings;
                    }
                }
            }
        }
    }
    texts.iter().map(|t| tfidf_embed(t)).collect()
}

pub fn is_fastembed_active() -> bool {
    #[cfg(feature = "fastembed")]
    {
        if let Some(mtx) = get_fastembed() {
            return mtx.lock().is_ok();
        }
    }
    false
}

// ─── TF-IDF Fallback Engine ──────────────────────────

fn get_synonyms(word: &str) -> Vec<&'static str> {
    match word {
        "login" | "signin" | "authenticate" => vec!["auth", "jwt", "session"],
        "auth" => vec!["login", "jwt", "session", "security"],
        "jwt" => vec!["auth", "token", "session"],
        "db" | "database" | "sql" => vec!["sqlite", "postgres", "supabase"],
        "ui" | "frontend" => vec!["components", "interface", "design"],
        "api" | "backend" => vec!["endpoints", "server", "routes"],
        "bug" | "error" | "fix" => vec!["issue", "patch", "problem"],
        "style" | "css" => vec!["tailwind", "styling", "design"],
        "perf" | "performance" => vec!["speed", "optimization", "fast"],
        "deploy" | "production" => vec!["hosting", "release", "cloudflare", "vercel"],
        _ => vec![],
    }
}

fn tfidf_embed(text: &str) -> Vec<f32> {
    let mut tokens = tokenize(text);

    let mut extra_tokens = Vec::new();
    for t in &tokens {
        for syn in get_synonyms(t) {
            extra_tokens.push(syn.to_string());
        }
    }
    tokens.extend(extra_tokens);

    if tokens.is_empty() {
        return vec![0.0; VECTOR_DIM];
    }

    let mut tf: HashMap<&str, f32> = HashMap::new();
    let total = tokens.len() as f32;
    for t in &tokens {
        *tf.entry(t.as_str()).or_default() += 1.0;
    }

    let mut vec = vec![0.0f32; VECTOR_DIM];
    for (term, count) in &tf {
        let freq = count / total;
        let idf = 1.0 + (1.0 / (term.len() as f32).sqrt());
        let weight = freq * idf;

        let h1 = hash_term(term, 0) % VECTOR_DIM;
        let h2 = hash_term(term, 1) % VECTOR_DIM;
        let h3 = hash_term(term, 2) % VECTOR_DIM;

        let sign1 = if hash_term(term, 3) % 2 == 0 { 1.0 } else { -1.0 };
        let sign2 = if hash_term(term, 4) % 2 == 0 { 1.0 } else { -1.0 };
        let sign3 = if hash_term(term, 5) % 2 == 0 { 1.0 } else { -1.0 };

        vec[h1] += weight * sign1;
        vec[h2] += weight * sign2 * 0.7;
        vec[h3] += weight * sign3 * 0.5;
    }

    for pair in tokens.windows(2) {
        let bigram = format!("{}_{}", pair[0], pair[1]);
        let h = hash_term(&bigram, 6) % VECTOR_DIM;
        let sign = if hash_term(&bigram, 7) % 2 == 0 { 1.0 } else { -1.0 };
        vec[h] += sign * 0.3;
    }

    normalize_vec(&mut vec);
    vec
}

// ─── Shared Utilities ──────────────────────────────

pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() { return 0.0; }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn rrf_score(bm25_rank: usize, vector_rank: usize) -> f64 {
    let k = 60.0;
    (1.0 / (k + bm25_rank as f64)) + (1.0 / (k + vector_rank as f64))
}

pub fn vec_to_blob(v: &[f32]) -> Vec<u8> {
    v.iter().flat_map(|f| f.to_le_bytes()).collect()
}

pub fn blob_to_vec(blob: &[u8]) -> Vec<f32> {
    blob.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn tokenize(text: &str) -> Vec<String> {
    text.to_lowercase()
        .split(|c: char| !c.is_alphanumeric() && c != '_' && c != '-')
        .filter(|w| w.len() >= 2)
        .map(String::from)
        .collect()
}

fn normalize_vec(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() { *x /= norm; }
    }
}

fn hash_term(term: &str, seed: u64) -> usize {
    let mut h: u64 = 14695981039346656037u64.wrapping_add(seed.wrapping_mul(6364136223846793005));
    for b in term.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(1099511628211);
    }
    h as usize
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_similar_texts() {
        let v1 = tfidf_embed("authentication login Supabase auth JWT");
        let v2 = tfidf_embed("user login authentication with JWT tokens");
        let v3 = tfidf_embed("CSS grid layout flexbox styling");
        let sim_related = cosine_similarity(&v1, &v2);
        let sim_unrelated = cosine_similarity(&v1, &v3);
        assert!(sim_related > sim_unrelated, "Related texts should have higher similarity");
    }

    #[test]
    fn test_blob_roundtrip() {
        let v = tfidf_embed("test embedding roundtrip");
        let blob = vec_to_blob(&v);
        let restored = blob_to_vec(&blob);
        assert_eq!(v.len(), restored.len());
        for (a, b) in v.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1e-7);
        }
    }
}
