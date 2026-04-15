# Prompt Cursor : Analyse comparative et amélioration de MemoryPilot

## Contexte

Je développe MemoryPilot, un système de mémoire persistante en Rust pour un assistant IA local (AURA). J'ai également installé Moltis (moltis-org/moltis), un agent IA personnel en Rust qui a son propre système de mémoire. Je veux comparer les deux et améliorer MemoryPilot en intégrant les meilleures fonctionnalités de Moltis.

## Fichiers à analyser

### MemoryPilot (notre code) : /Users/antoinepinelli/Cursor/App/MemoryPilot/src/
- db.rs (4756 lignes) : base de données principale, stockage, recherche BM25 + TF-IDF
- graph.rs (239 lignes) : knowledge graph (triples subject-predicate-object)
- embedding.rs (89 lignes) : embeddings vectoriels
- tools.rs (792 lignes) : outils MCP exposés (add, search, recall, etc.)
- gc.rs (171 lignes) : garbage collection des mémoires expirées
- watcher.rs (199 lignes) : surveillance de fichiers
- http.rs (230 lignes) : serveur HTTP
- main.rs (228 lignes) : point d'entrée
Total : 6750 lignes

### Moltis Memory (référence) : /tmp/moltis-src/crates/memory/src/
- store_sqlite.rs (769 lignes) : stockage SQLite avec embeddings
- manager.rs (1170 lignes) : gestionnaire principal, auto-compaction, contexte
- search.rs (448 lignes) : recherche hybride vectorielle + full-text
- reranking.rs (337 lignes) : re-ranking des résultats de recherche
- tools.rs (716 lignes) : outils MCP pour l'agent
- embeddings_local.rs (181 lignes) : embeddings locaux GGUF
- embeddings_batch.rs (322 lignes) : batch embeddings via API
- embeddings_openai.rs (194 lignes) : embeddings OpenAI
- embeddings_fallback.rs (283 lignes) : fallback entre providers
- session_export.rs (367 lignes) : export de sessions
- watcher.rs (182 lignes) : file watching avec live sync
- splitter.rs (254 lignes) : text splitting pour chunking
- chunker.rs (153 lignes) : chunking intelligent
- writer.rs (135 lignes) : écriture mémoire
- config.rs (109 lignes) : configuration
- contract.rs (172 lignes) : contrats/interfaces
Total : 6045 lignes

## Tâches demandées

### 1. Analyse comparative détaillée

Lis les deux codebases en entier et fais une comparaison fonction par fonction :

| Fonctionnalité | MemoryPilot | Moltis Memory | Qui est meilleur | Ce qui manque à MemoryPilot |
|---|---|---|---|---|
| Stockage | SQLite BM25 + TF-IDF | SQLite + embeddings vectoriels | ? | ? |
| Recherche | BM25 hybrid | Hybrid vector + full-text + reranking | ? | ? |
| Embeddings | Basique | Local GGUF + OpenAI + batch + fallback | ? | ? |
| Auto-compaction | Non | Oui (95% context window) | ? | ? |
| Chunking | Non | Splitter + chunker intelligent | ? | ? |
| Reranking | Non | Oui (337 lignes dédiées) | ? | ? |
| File watching | Basique | Live sync | ? | ? |
| Session export | Non | Oui | ? | ? |
| Knowledge graph | Oui (triples) | Non | ? | ? |
| Garbage collection | Oui | Non | ? | ? |
| MCP tools | Oui | Oui | ? | ? |

### 2. Fonctionnalités à intégrer dans MemoryPilot (priorité haute)

En te basant sur l'analyse, identifie les 5 fonctionnalités les plus importantes de Moltis qui manquent à MemoryPilot et implémente-les :

1. **Auto-compaction** : quand la mémoire atteint un seuil (ex: 95% du context window), résumer automatiquement les anciennes mémoires pour libérer de l'espace tout en gardant l'essentiel. Regarde comment manager.rs de Moltis fait ça.

2. **Reranking** : après la recherche BM25/vectorielle, re-classer les résultats par pertinence contextuelle. Regarde reranking.rs de Moltis et intègre une version dans notre search.

3. **Chunking intelligent** : quand on stocke un gros texte (conversation longue, article web, PDF), le découper en chunks sémantiques au lieu de le stocker en un bloc. Regarde splitter.rs et chunker.rs de Moltis.

4. **Embeddings multi-provider avec fallback** : supporter les embeddings locaux (GGUF via Ollama), OpenAI en batch, et un fallback automatique. Regarde embeddings_fallback.rs de Moltis.

5. **Session export** : pouvoir exporter les mémoires d'une session en markdown/JSON pour backup ou partage. Regarde session_export.rs de Moltis.

### 3. Garder nos avantages

MemoryPilot a des fonctionnalités que Moltis n'a PAS. Ne les supprime pas :

- **Knowledge Graph** (graph.rs) : triples subject-predicate-object avec timeline. C'est unique et puissant pour les relations entre entités.
- **Garbage Collection** (gc.rs) : nettoyage automatique des mémoires expirées.
- **Projets multiples** : MemoryPilot gère 8 projets séparés. Moltis n'a pas cette notion.

### 4. Architecture cible

Le résultat final doit être un MemoryPilot qui combine le meilleur des deux :

```
MemoryPilot v4.0
  |
  +-- Stockage SQLite (existant)
  +-- Recherche hybrid BM25 + Vector + Reranking (amélioré)
  +-- Knowledge Graph avec triples (existant, unique)
  +-- Auto-compaction intelligente (nouveau, inspiré Moltis)
  +-- Chunking sémantique (nouveau, inspiré Moltis)
  +-- Embeddings multi-provider + fallback (nouveau, inspiré Moltis)
  +-- Session export markdown/JSON (nouveau, inspiré Moltis)
  +-- File watching avec live sync (amélioré)
  +-- Garbage collection (existant)
  +-- Projets multiples (existant, unique)
  +-- HTTP API + MCP tools (existant)
```

### 5. Contraintes techniques

- Langage : Rust uniquement
- Pas de dépendances lourdes (pas de Python, pas de Node)
- Compatible avec Ollama pour les embeddings locaux (nomic-embed-text)
- Le code doit compiler avec `cargo build --release`
- Garder la rétro-compatibilité avec l'API MCP existante
- Le code source de Moltis memory est dans /tmp/moltis-src/crates/memory/src/

### 6. Livrable

Commence par l'analyse comparative complète (tableau), puis implémente les 5 fonctionnalités dans l'ordre de priorité. Fais un commit par fonctionnalité avec un message clair.
