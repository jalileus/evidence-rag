# Доказательный RAG по ТФКП

**ФИО:** Алхажали Жалал

**Группа:** М05-607в

**Задание:** доказательный Retrieval-Augmented Generation (RAG) по учебнику «Лекции по ТФКП» (Горяйнов, Половинкин, 2025).

## Описание

Проект реализует систему ответов на вопросы по теории функций комплексного переменного (ТФКП). Ответы должны опираться на предоставленный учебный материал: модель выбирает допустимые варианты, приводит краткое обоснование и указывает источники через исходные `block_id`. Если доступных фрагментов недостаточно для обоснованного ответа, система должна воздержаться от ответа.

Используемая модель: **`Qwen/Qwen3.5-4B`**.

## Структура Проекта

```text
.
├── predict.py
├── build_index.py
├── evaluate.py
├── config.json
├── runtime.json
├── requirements.txt
├── experiments.csv
├── dev_answers.jsonl   # = /data/dev_answers_final.jsonl
├── dev_trace.jsonl     # = /data/dev_trace_final.jsonl
├── own_questions.jsonl
├── own_gold.jsonl     
├── .gitignore
├── report.md
├── Readme.md
├── data/
│   ├── native_text.md
│   ├── corpus.jsonl
│   ├── dev_questions.jsonl
│   ├── dev_gold.jsonl
│   ├── own_questions.jsonl
│   ├── own_gold.jsonl
│   ├── dev_answers_<experiment>.jsonl
│   ├── dev_trace_<experiment>.jsonl
│   ├── own_answers_final.jsonl
│   └── own_trace_final.jsonl
├── scripts/
│   ├── extract_native_markdown.py
│   ├── extract_text.py
│   └── extract_questions_dev.py
└── collab/
    └── pipline_hw.ipynb
    └── pipline_hw.html
```

Файлы `dev_answers_<experiment>.jsonl` и `dev_trace_<experiment>.jsonl` соответствуют отдельным экспериментам.

## Подготовка Корпуса и Вопросов

Вопросов и разметки были подготовлены в рамках проекта.

Последовательность подготовки:
1. `scripts/extract_native_markdown.py` извлекает Markdown-текст из учебника и формирует `native_text.md`.
2. `scripts/extract_text.py` преобразует Markdown в `data/corpus.jsonl`. Корпус содержит текстовые фрагменты, формулы и исходные идентификаторы `block_id`.
3. `scripts/extract_questions_dev.py` формирует `data/dev_questions.jsonl` и `data/dev_gold.jsonl` на основе `tfkp_15_qa.md` и корпуса, используя фрагменты корпуса для разметки источников.

**нужно будет скачать tfkp_15_qa.md и учебнику (textbook.pdf) в `data`**

В проект также включены собственные вопросы и соответствующая разметка: `own_questions.jsonl` и `own_gold.jsonl`.

## Индексирование

Для поиска используется BM25 (`BM25Okapi`). Индекс строится по `data/corpus.jsonl`; параметры токенизации и поиска заданы в `config.json`.

Перед запуском предсказания необходимо создать индекс с помощью `build_index.py`. Например:

```bash
python build_index.py --corpus ./data/corpus.jsonl --output ./index
```



## Запуск Эксперимента

Установите зависимости:

```bash
pip install -r requirements.txt
```

Убедитесь, что модель `Qwen/Qwen3.5-4B` доступна и настроен требуемый для `predict.py` совместимый API генерации. В проектном ноутбуке `collab/pipline_hw.ipynb` использовалась среда Google Colab с GPU **NVIDIA T4**.

Пример запуска одного эксперимента:

```bash
python predict.py \
  --experiment-name rag_single \
  --index ./index \
  --questions ./data/dev_questions.jsonl \
  --output ./data/dev_answers_rag_single.jsonl \
  --trace ./data/dev_trace_rag_single.jsonl \
  --config ./config.json \
  --runtime ./runtime.json
```

Чтобы воспроизвести другие режимы, замените `rag_single` на соответствующее имя эксперимента и задайте отдельные файлы результатов и трассировки.
так же нужно будет заменить "name" в `runtime.json`

Поддерживаемые режимы:

| Режим                | Описание                                                                             |
| -------------------- | ------------------------------------------------------------------------------------ |
| `no_context`         | Без поиска по корпусу; модель отвечает без извлечённого контекста                    |
| `rag_single`         | BM25 и один вызов LLM, без повторной попытки                                         |
| `final`              | Основная RAG-система с повторной попыткой при некорректных цитатах                   |
| `ablation_no_rerank` | Абляция ранжирования: фрагменты берутся в порядке корпуса, а не по оценке BM25       |
| `ablation_top1`      | Абляция размера контекста: используется только один извлечённый фрагмент             |

В режимах, где предусмотрена повторная попытка, она может использоваться для исправления некорректных цитат. `rag_single` намеренно не выполняет повторный вызов.

## Оценивание

`evaluate.py` оценивает ответы с использованием предсказаний, трассировок и эталонной разметки `dev_gold.jsonl`. Результаты экспериментов собраны в `experiments.csv`.

Пример вызова:

```bash
python evaluate.py --gold ./data/dev_gold.jsonl --answers-dir data --traces-dir data
```

## Конфигурация

`config.json` задаёт параметры BM25 и токенизации: приведение к нижнему регистру, обработку математических символов, минимальную длину токена, русский язык и стемминг.

`runtime.json` задаёт режим эксперимента, размер выдачи (`top_k = 5`), параметры генерации (`temperature = 0.0`, `max_tokens = 1024`) и тайм-аут в 180 секунд.

## Ресурсы и Воспроизводимость

- Модель: `Qwen/Qwen3.5-4B`.
- Среда запуска модели в Colab: бесплатный GPU NVIDIA T4.
- Поиск: BM25 на CPU; характеристики индексирования приведены в отчёте.
- [Горяйнов, Половинкин. «Лекции по ТФКП», 2025](https://new.mipt.ru/upload/mathematics/study/uchebniki/%D0%9B_%D0%A2%D0%A4%D0%9A%D0%9F_%D0%93%D0%BE%D1%80%D1%8F%D0%B9%D0%BD%D0%BE%D0%B2_%D0%9F%D0%BE%D0%BB%D0%BE%D0%B2%D0%B8%D0%BD%D0%BA%D0%B8%D0%BD_2025.pdf)
