# Full-data experiment: Phi-3 Text-to-SQL QLoRA

Completed on Kaggle with two Tesla T4 GPUs; evidence published 31 October 2026. Training source: `613a4b9291ab8e67b88d07f5e15b35486cb53091`. Base revision: `f39ac1d28e925b323eae81227eaba4464caced4e`.

## Protocol

All 7,000 Spider training rows were considered; 82 exceeded 2,048 tokens, leaving 6,918. Original database-separated splits were preserved. Validation used 1,034 rows; paired generation evaluated the first 200 validation questions. No training-subset flag was used. Seed 42; two configured epochs; reported epoch 1.998265; 864 optimizer steps. DDP used one NF4 model replica per GPU, batch 1 per device and accumulation 8 (effective batch 16). Completion-only loss masks prompts and padding.

The lowest validation loss was 0.235159 at checkpoint 300. Trainer reloaded this checkpoint after completing training. The final exported adapter is byte-identical to checkpoint 300, rather than the last checkpoint. Thus “two epochs” describes the completed training schedule, not the selected adapter's training age.

Base and adapter receive identical prompts, greedy decoding, batch size 4 and a 150-token generation cap. Prompt-file SHA256: `f8374119dfd3243bc73e2e5fd1d91a3effdc931d9ca115e74284653f99305cac`. Original reports and independent CPU replay are published under `reports/full-data/`; replay reproduced every aggregate metric.

## Results

| Metric | Base Phi-3 | QLoRA adapter | Change |
| --- | ---: | ---: | ---: |
| Normalized exact match | 14.5% | 23.0% | +8.5 pp |
| SQLite execution accuracy | 50.0% | 59.0% | +9.0 pp |
| BLEU-4 (0–1) | 0.3076 | 0.6979 | +0.3903 |
| BERTScore | Not measured | Not measured | Skipped |

| Paired execution outcome | Questions |
| --- | ---: |
| Both correct | 88 |
| Base error fixed | 30 |
| Base-correct prediction regressed | 12 |
| Both incorrect | 70 |

The adapter adds 18 correct executions (118 versus 100). All gold queries executed successfully. Training loop: 16,191.20 seconds (4.50 hours); training stage including loading/saving: 16,274.90 seconds (4.52 hours). Rank-zero peak allocation: 4.35 GiB; this is not total memory across GPUs. Mean training loss: 0.110036. Adapter parameters: 25,165,824; base weights are not bundled.

## Representative cases

These are the first matching cases in stored order, selected by execution outcome rather than presentation quality.

### Base error fixed — example 6, `concert_singer`

What is the average, minimum, and maximum age for all French singers?

Reference:
```sql
SELECT avg(age) ,  min(age) ,  max(age) FROM singer WHERE country  =  'France'
```

Base:
```sql
SELECT AVG(s.Age) AS Average_Age, MIN(s.Age) AS Minimum_Age, MAX(s.Age) AS Maximum_Age
FROM singer s
JOIN singer_in_concert sic ON s.Singer_ID = sic.Singer_ID
JOIN concert c ON sic.concert_ID = c.concert_ID
JOIN stadium st ON c.Stadium_ID = st.Stadium_*
WHERE s.Country = 'France';
```

Adapter:
```sql
SELECT avg(age),  min(age),  max(age) FROM singer WHERE country  =  'France'
```

### Adapter regression — example 24, `concert_singer`

For each stadium, how many concerts play there?

Reference:
```sql
SELECT T2.name ,  count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.stadium_id
```

Base:
```sql
SELECT s.Name, COUNT(c.concert_ID) AS Number_of_Concerts
FROM stadium s
JOIN concert c ON s.Stadium_ID = c.Stadium_ID
GROUP BY s.Name;
```

Adapter:
```sql
SELECT T1.name,  count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.name
```

### Both incorrect — example 7, `concert_singer`

Show the name and the release year of the song by the youngest singer.

Reference:
```sql
SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1
```

Base:
```sql
SELECT Name, Song_release_year
FROM singer
ORDER BY Age ASC
LIMIT 1;
```

Adapter:
```sql
SELECT name,  song_release_year FROM singer ORDER BY age ASC LIMIT 1
```

## Interpretation and limits

These are custom development metrics, not an official Spider test score or proof of SQL equivalence. Raw outputs are scored without stripping prose or Markdown. Single-database equality can reward accidentally matching empty results; projection order matters. The 200 questions are the first 200 validation rows, include paraphrases, and are not independent observations. Validation loss also selected the checkpoint, so these are not untouched test results. Only one seed was run; no statistical-significance claim is made.

The earlier pilot used a shuffled 200-row cohort, so its 57.5% adapter score is not directly comparable with this run's 59%. See [pilot analysis](PILOT.md). The previous single-GPU full-data run took 8.25 hours for the training stage; the rerun took 4.52 hours (observed 1.82× reduction). This is a comparison of separate runs, not a controlled scaling benchmark. The previous full-data adapter was not retained; its notebook is archived as an unretained run, not the released model.

Next experiments: multiple seeds, output-format extraction applied symmetrically, rank/learning-rate ablations, and an untouched official test-suite evaluation. Inspect regressions before claiming generalization.

Spider attribution: Yu et al., EMNLP 2018, CC BY-SA 4.0, https://yale-lily.github.io/spider.
