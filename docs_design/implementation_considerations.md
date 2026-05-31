# Что Стоит Держать В Голове

Этот документ фиксирует практические оговорки по текущей реализации
Confluence -> Qdrant ingestion pipeline.

## Большие Confluence Spaces

В текущем pipeline есть потенциально лишний API-проход: в
`rague/ingestion/confluence_to_qdrant.py` сначала вызывается
`discover_page_ids()`, чтобы посчитать и записать в changelog количество
найденных страниц, а затем `lazy_load()` внутри loader-а снова вызывает
discovery и заново получает список страниц.

Для маленьких spaces это не критично. Для больших spaces это может стать
заметной проблемой по времени, rate limits и нагрузке на Confluence.

### Практические Критерии Большого Space

Space стоит считать большим, если выполняется хотя бы одно условие:

- в scope больше 500-1000 страниц;
- дерево от `parent_page_id` глубже 4-5 уровней и содержит много sibling-страниц;
- один ingestion run занимает больше 2-5 минут только на discovery;
- Confluence API начинает возвращать throttling, timeout или нестабильные ответы;
- количество страниц требует десятков paginated-запросов;
- загрузка запускается часто, например по расписанию каждые 5-15 минут;
- space содержит много вложений, потому что attachments умножают количество
  документов для обработки.

Для MVP можно считать порогом внимания примерно 500 страниц. Начиная с этого
размера лучше убрать повторный discovery и передавать уже найденный список
страниц дальше в загрузку.

### Что Улучшить Позже

Для больших spaces лучше сделать один из вариантов:

1. `discover_page_ids()` вызывается один раз, а `load_pages(page_ids)` принимает
   уже найденный список.
2. `lazy_load()` принимает optional `page_ids`, чтобы не запускать discovery
   повторно.
3. Discovery возвращает не только IDs, а lightweight page metadata:
   `page_id`, `title`, `version`, `source_updated_at`, `path`.
4. Для `space_key` основной механизм discovery должен быть CQL по
   `lastmodified`, а рекурсивный обход дерева лучше использовать для enrichment
   metadata и path.

## Attachments

Сейчас attachments концептуально учтены в плане, но в runtime pipeline не
индексируются. Текущая версия загружает страницы Confluence как Markdown, режет
их на chunks и пишет chunks страниц в Qdrant. Вложения страниц сохраняются
локально как образцы, но не конвертируются и не записываются в Qdrant.

### MVP Политика

Если при обходе Confluence встречается attachment, в текущем MVP мы не создаем
из него `Document`, не режем на chunks и не пишем в Qdrant.

Вместо этого сохраняем attachment локально как образец для будущей разработки
конвертеров. Сохраняем максимум один файл каждого расширения.

Пример:

```text
sample.doc
sample.docx
sample.pdf
sample.pptx
```

Если встретились файлы с расширениями `doc`, `docx`, `pdf`, `pptx`, локально
должно оказаться четыре файла - по одному на каждое расширение. Следующие
attachments с уже сохраненным расширением пропускаются.

Цель этой политики - собрать минимальный набор реальных примеров форматов, не
засоряя Qdrant непроверенным binary/OCR-контентом и не раздувая локальное
хранилище.

Рекомендуемое локальное расположение для таких образцов:

```text
data/attachment_samples/
```

Эта директория должна оставаться вне git, потому что attachments могут содержать
внутренние данные.

### Как Их Нужно Обрабатывать

В будущей версии attachments должны стать отдельными logical documents, но
наследовать metadata страницы, на которой они приложены.

Пример модели:

```text
Confluence page
  -> page Document
  -> page chunks

Confluence attachment
  -> attachment Document
  -> attachment chunks
```

В будущей Qdrant-модели записью все равно остается chunk. То есть attachment,
как и страница, будет разбиваться на chunks, и каждый chunk будет писаться
отдельным point.

### Metadata Для Attachments

Attachment должен наследовать page metadata:

- `source_type`
- `space`
- `page_id`
- `parent_page_id`
- `parent_page_ids`
- `parent_titles`
- `ancestors`
- `path`
- `source`

И иметь собственные поля:

- `document_type = "attachment"`;
- `document_id = "confluence:page:{page_id}:attachment:{attachment_id}"`;
- `attachment_id`;
- `attachment_title`;
- `attachment_filename`;
- `attachment_media_type`;
- `attachment_version`;
- `attachment_updated_at`;
- `source_updated_at`, равный `attachment_updated_at` для attachment-документа;
- `chunk_id = "{document_id}:v{attachment_version}:chunk:{chunk_index}"`.

Без собственных attachment identifiers будет сложно корректно обновлять,
удалять и дедуплицировать вложения.

### Loader Или Converter?

Для attachments лучше разделить ответственность:

- Confluence source layer получает список attachments и скачивает bytes или file
  metadata.
- Attachment converters превращают конкретные форматы в Markdown `Document`.
- Ingestion workflow обрабатывает page documents и attachment documents
  одинаково: split -> embed -> delete old chunks -> upsert new chunks.

То есть attachment converters могут быть оформлены как loaders/parsers для
конкретных типов файлов, но не стоит перегружать Confluence page loader
логикой PDF/DOCX/XLSX/OCR.

### Будущая Политика Индексации

Когда появятся converters, включать индексирование attachments стоит явным
флагом:

```text
index_attachments = true
```

Начинать лучше с ограниченного набора типов:

- Markdown/text;
- PDF с текстовым слоем;
- DOCX;
- позже XLSX и OCR.

### Риски Attachments

Основные риски:

- большие файлы резко увеличат время ingestion;
- один attachment может породить больше chunks, чем сама страница;
- binary/OCR-конвертация может быть нестабильной;
- вложения часто содержат таблицы, где обычный Markdown conversion может терять
  структуру;
- права доступа к attachment могут отличаться от прав доступа к странице;
- attachment может обновиться без изменения текста страницы.

Поэтому даже MVP-сбор образцов attachments должен иметь отдельную статистику в
changelog:

- attachments discovered;
- attachment samples saved;
- attachments skipped;
- attachment sample extensions;
- unsupported attachment types;
- attachments failed.

## Текущий Вывод

Двойной discovery не блокирует MVP, если scope маленький или средний. Но для
больших spaces лучше быстро перейти к single-discovery flow.

Attachments в текущем MVP не пишем в Qdrant. Мы только сохраняем локальные
образцы по одному файлу каждого расширения, чтобы позже на реальных примерах
реализовать converters. Когда индексирование attachments будет включено, их
лучше рассматривать как отдельные documents, связанные со страницей через
metadata. Это сохранит управляемые update/delete, нормальные citations и
возможность фильтровать результаты по типу документа.
