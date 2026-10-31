# Pilot experiment: Phi-3 Text-to-SQL QLoRA

Run on 30 October 2026 using one Tesla T4 in Kaggle. Training source: `a2c83fc312eba6b805a435615dbae25ea457802f`. Scoring source: `1afb4185fa4d0a2cd35d28fb72c52a233929e01f`. Base model revision: `f39ac1d28e925b323eae81227eaba4464caced4e`.

## Question

Does a small completion-only adapter improve the base model's ability to return executable SQL for unseen Spider database schemas?

## Protocol and scope

A seed-42 shuffle selected 1,000 rows from the original 7,000-row training split and 200 rows from the separate 1,034-row validation split. Thirteen training rows exceeded the 2,048-token limit, leaving 987. The one-epoch configuration completed 61 optimizer steps (reported epoch fraction 0.988855). No validation score was used to choose between multiple runs.

The base and adapter received identical prompts, greedy decoding, a 150-token output cap, and batch size 4. Predictions were saved and then rescored on CPU using the published evaluator. The prompt-file SHA-256 is `d02e6d6dfce1bc0c53e7d425fa2f672f735b7ec50f432e2bb836027dc8be3171`. Original and replay reports are both preserved.

## Results

| Metric | Base Phi-3 | QLoRA adapter | Change |
| --- | ---: | ---: | ---: |
| Normalized exact match | 11.0% | 26.0% | +15.0 pp |
| SQLite execution accuracy | 53.0% | 57.5% | +4.5 pp |
| BLEU-4 (0–1) | 0.3878 | 0.6547 | +0.2669 |
| BERTScore | Not measured | Not measured | Skipped |

| Paired execution outcome | Questions |
| --- | ---: |
| Both correct | 85 |
| Adapter fixes a base error | 30 |
| Adapter regresses a base-correct case | 21 |
| Both incorrect | 64 |

Base predictions produced 41 execution errors; adapter predictions produced 32. All gold queries executed successfully. Execution accuracy is strict result equality on one database, not a formal semantic-equivalence proof. Empty results can match accidentally, and column-order differences count as mismatches. The net 4.5-percentage-point gain on this small development sample is preliminary; it is not claimed to be statistically significant.

## Representative examples

The examples below are the first matching cases in the stored evaluation order, not selected for presentation quality. 'Correct' here means the custom execution metric matched the reference result.

### A base error fixed — example 5, `world_1`

What are the names of the countries that are in the continent of Europe and have a population of 80000?

Reference:
```sql
SELECT Name FROM country WHERE continent  =  "Europe" AND Population  =  "80000"
```

Base:
```sql
SELECT Name FROM country WHERE Continent = 'Europe' AND Population = 80 ISBN:

SELECT Name FROM country WHERE Continent = 'Europe' AND Population = 80000;
```

Adapter:
```sql
SELECT Name FROM country WHERE Continent  =  'Europe' AND Population  =  80000
```

### An adapter regression — example 24, `world_1`

What are the countries where either English or Dutch is the official language ?

Reference:
```sql
select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "english" and isofficial  =  "t" union select t1.name from country as t1 join countrylanguage as t2 on t1.code  =  t2.countrycode where t2.language  =  "dutch" and isofficial  =  "t"
```

Base:
```sql
SELECT DISTINCT country.Name
FROM country
JOIN countrylanguage ON country.Code = countrylanguage.CountryCode
WHERE countrylanguage.Language IN ('English', 'Dutch') AND countrylanguage.IsOfficial = 'Yes';
```

Adapter:
```sql
SELECT T2.Name FROM countrylanguage AS T1 JOIN country AS T2 ON T1.CountryCode  =  T2.Code WHERE T1.Language  =  "English" OR T1.Language  =  "Dutch"
```

### A case both missed — example 1, `wta_1`

How many players are from each country?

Reference:
```sql
SELECT count(*) ,  country_code FROM players GROUP BY country_code
```

Base:
```sql
SELECT country_code, COUNT(player_id) AS num_players
FROM players
GROUP BY country_code;
```

Adapter:
```sql
SELECT country_code,  count(*) FROM players GROUP BY country_code
```

## Interpretation and next experiments

The pilot improved the specified raw-SQL metric and output-format consistency, while retaining substantial semantic errors and introducing regressions. The base sometimes includes Markdown or extra text, which the raw-output protocol penalizes. A useful next ablation is to apply the same code-fence extraction to both models, then compare semantic errors separately from formatting errors. Other next steps are a larger training split, multiple seeds, rank/learning-rate ablations, and the official Spider test-suite evaluator on an untouched evaluation set.

Single-database equality can give misleading successes. For example, a gold condition and an incorrect generated condition can both return no rows. Inspecting the stored queries is essential; an execution-match label alone does not prove the query correctly answers the question.

Spider-derived examples and report content retain the dataset's CC BY-SA 4.0 attribution: Yu et al., EMNLP 2018, https://yale-lily.github.io/spider.
