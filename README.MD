# "Умные покупки от МТС" 

## Команда "Arbejdsglaede"

### Проблема

Повышение количества транзакций (покупок) пользователей и категорий, в которых они закупаются. 
Провести пользователя по цепочке "купил порошок - купи стиральную машину - нужна одежда - нужны иголки"

### Решение

Расширение для мобильного приложения по персональным рекомендациям

### Используемые модели и решения

1. CatBoost
2. K-means
3. Flask 
4. SQLite

# Запуск проекта


```python
git clone https://github.com/dead142/mts-ai.git

cd <project>

pip install requirements.txt 

python app.py

```

## Примеры запросов  (PyCharm)
```
POST http://127.0.0.1:81/post
Content-Type: application/json

{"MCC_CODE": 2000, "SUM_TRANS": 100}

###

POST http://127.0.0.1:81/post
Content-Type: application/json

{"MCC_CODE": 8000, "SUM_TRANS": 100000}

###

POST http://127.0.0.1:81/post
Content-Type: application/json

{"MCC_CODE": 1, "SUM_TRANS": 1}
 ```

## Примеры запросов (curl)
```
curl -X POST --location "http://127.0.0.1:81/post" \
    -H "Content-Type: application/json" \
    -d "{\"MCC_CODE\": 8000, \"SUM_TRANS\": 100000}"
```

## Отчет сервера (товары категорий)

- id
- name
- price
- category
```json
[[3, \"\Т\о\в\а\р 13\", 100000, 1], [6, \"\Т\о\в\а\р 22\", 100000, 2], [9, \"\Т\о\в\а\р 32\", 1000000, 3]]

```
