## Виртуальная машина

В данном репозитории (почти) реализована виртуальная машина, исполняющую байткод питона. На питоне. Мной из всего этого репо написан только vm.py, 
остальное любезно предоставили лекторы ШАДА

### Как запустить конкретный тест-кейс

Обратите внимание на закомментированный код в файле `vm_runner.py`, он поможет вам при дебаге.

```bash
$ pytest test_public.py::test_all_cases[simple] -vvv
```
Для тех у кого `zsh`
```bash
$ pytest test_public.py::test_all_cases\[simple\] -vvv
```

### Как запустить все тесты

```bash
$ pytest test_public.py -vvv --tb=no
```

### Как посмотреть распределение операций по тестам

```bash
$ pytest test_stat.py -s
```

### Очень полезные ссылки

* Документация к dis: https://docs.python.org/release/3.11.5/library/dis.html#module-dis. Там описаны все существующие операции байткода питона.
* Академический проект интерпретатора для PY27 и PY33, снабженный множеством комментариев, но не лишенный проблем: https://github.com/nedbat/byterun.
Его детальное обсуждение в блоге: http://www.aosabook.org/en/500L/a-python-interpreter-written-in-python.html.
* Исходный код интерпретатора CPython - поможет разобраться с тонкостями: https://github.com/python/cpython/blob/3.11/Python/ceval.c.
