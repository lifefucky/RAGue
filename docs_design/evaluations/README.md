# Evaluation Iterations

Каталог хранит записи evaluation-итераций: после каждой попытки улучшить метрики
создаётся отдельный Markdown-файл с конфигурацией решения и результатами прогона.

Цель — накапливать историю экспериментов, пока качество ответа не станет
приемлемым.

## Naming Convention

```
{NNN}_{short-description}.md
```

- `NNN` — порядковый номер итерации, три цифры с ведущими нулями (`001`, `002`, …);
- `short-description` — kebab-case, кратко описывает решение или отличие от
  предыдущей итерации.

Примеры:

- `001_baseline-hybrid-agentic-rag.md` — первая baseline-конфигурация;
- `002_ms-marco-reranker-baseline.md` — смена reranker относительно `001`;
- `003_chunk-size-800-russian-bm25.md` — изменение chunking и BM25 tokenizer.

## When To Create A New File

Новый файл создаётся после каждого evaluation run, который меняет хотя бы одно из:

- embedding или reranking модель;
- retrieval/agent/chunking/ingestion параметры;
- prompt-ы или workflow logic;
- evaluation dataset или methodology.

Если прогон повторяет предыдущую конфигурацию без изменений, отдельный файл не
нужен — достаточно дополнить существующую запись.

## Required Sections

Каждый файл итерации должен содержать:

1. **Summary** — одно-два предложения: что тестировали и зачем.
2. **Changes From Previous Iteration** — diff относительно `NNN-1`; для `001` —
   «baseline, предыдущей итерации нет».
3. **Configuration** — модели, env defaults, архитектурные паттерны, затронутые
   компоненты.
4. **Evaluation Setup** — dataset, scope корпуса, команды запуска, версия кода
   или commit (если есть).
5. **Evaluation Results** — таблицы метрик и краткие qualitative notes; пустые
   поля допустимы только с явной пометкой «не измерялось».
6. **Conclusion** — приемлемо ли качество, что улучшилось/ухудшилось.
7. **Next Step** — гипотеза для следующей итерации.

Метрики и пороги — см. [`metrics.md`](../metrics.md).

## Index

| Iter | File | Summary | Quality OK |
| --- | --- | --- | --- |
| 001 | [`001_baseline-hybrid-agentic-rag.md`](001_baseline-hybrid-agentic-rag.md) | Baseline: E5 + hybrid BM25/vector + bge reranker + agentic workflow + citations | no — formal eval pending |
| 002 | [`002_step-4-evaluation-baseline.md`](002_step-4-evaluation-baseline.md) | Step 4 evaluation infrastructure, dataset, runners, CLI, opt-in benchmarks | no — live corpus metrics pending |
