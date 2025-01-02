## Проект «Система классификации сигналов ЭКГ для определения состояния здоровья человека»

<b>Цель проекта:</b> Разработать систему классификации физиологических сигналов (ЭКГ). Результат годового проекта — веб-сервис/бот для определения состояния здоровья человека по электрокардиограмме.

### Команда 

| ФИО              | Telegram | Github |
| :---------------- | :---------------- | :---------------- |
| Герберт Лика Сергеевна       |   [@akiltrebreg](https://t.me/akiltrebreg)   | akiltrebreg |
| Завадский Никита Павлович           |   [@metand](https://t.me/metand)   | farcon1 |
| Кузнецова Анна Сергеевна    |  [@Kuznetsova_anse](https://t.me/Kuznetsova_anse)   | KuzetsovaAnn |
| Смирнов Иван Владимирович |  [@sourcreamcake](https://t.me/sourcreamcake)   | ismirn |

### Кураторы

| ФИО              | Telegram | Github |
| :---------------- | :---------------- | :---------------- |
| Карагодин Никита Сергеевич       |   [@Einstein_30](https://t.me/Einstein_30)   | NickKar30 |
| Паточенко Евгений Анатольевич           |   [@evg_pat](https://t.me/evg_pat)   | evgpat |

### Содержимое проекта
`dataset.md` - подробное описание используемых в проекте данных <br>
`eda_notebook.ipynb` - ноутбук со сборкой датасета и разведочным анализом данныхх <br>
`EDA.md` - выводы о проведенном разведочном анализе данных

## Лицензия
Проект выполняется под **лицензией MIT**.

# Деплой проекта и демонстрация работы

## Адрес проекта
http://195.133.13.244:8501/

## Описание хода деплоя

### 1. Аренда VPS-сервера
Вы можете развернуть проект и на своей локальной машине, однако в ходе нашей работы мы арендовали VPS-сервер

### 2. Настройка сервера
Так как это пошаговое описание работы, то здесь будет рассказ про настройку сервера и последующий деплой, вы можете руководствоваться им или же своими собственными соображениями (в этом случае, полноценная работа проекта не гаранстируется)

#### 1) судо
Настраиваем наш пакетный менеджер
```bash
$ sudo apt update
$ sudo apt upgrade
```
#### 2) юзер (опционально)
Не рекомендуется в целях всего и всея работать в линуксе из под рута - для этого добавим нового пользователя и сделаем его судоером
```bash
$ sudo adduser www
$ sudo usermod -aG sudo www
```
Перейдем на этого юзера
```bash
$ sudo su - www
```
#### 3) докер
Теперь нам нужно установить докер и docker-compose, ну и гит, если его нет
```bash
$ sudo apt update
$ sudo apt install docker.io
$ sudo systemctl start docker
$ sudo systemctl enable docker

$ sudo apt install docker-compose

$ sudo apt install git
```
#### 4) клонируем репозиторий
Для этого сначала давайте перейдем в домашнюю директорию пользователя `www`
```bash
$ cd /home/www
```
Теперь клонируем репозиторий с проектом
```bash
$ git clone https://github.com/akiltrebreg/ai24_ecg.git
```
#### 5) немного докер-фокусов и почти готово
Заходим в папку с проектом
```bash
$ cd ai24_ecg
```
Запускаем  docker-compose
```bash
$ sudo docker-compose up -d
```
Будет происходить примерно следующее:
![докер-фокусы](https://storage.yandexcloud.net/ecg-project/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202024-12-30%20%D0%B2%2014.46.36.png)
Если и у вас все получится - ваш проект булет лежать по адресу http://localhost:8085/
где вместо localhost - ip-адрес вашего сервера, если вы как и мы развернули его в некой сети Интернет
#### 6) P.S. С какими проблемами мы столкнулись в вопросах деплоя?
Как только проект был развернут, первые пробы обучения моделей заканчивались крахом - наш прод валился:
![провал](https://storage.yandexcloud.net/ecg-project/%D0%A1%D0%BD%D0%B8%D0%BC%D0%BE%D0%BA%20%D1%8D%D0%BA%D1%80%D0%B0%D0%BD%D0%B0%202024-12-30%20%D0%B2%2015.27.05.png)
Недолго думая, мы увеличили количество оперативной памяти на сервере, и. о чудо - все стало работать стабильно
Поэтому, 4Гб оперативной памяти - минимальное требование нашего проекта (если конечно проектом не будут пользоваться ~~все ассистенты курса "Инструменты разработки"~~ очень много людей)

### Демо
Пройдя по адресу http://195.133.13.244:8501/ мы оказываемся на тсранице стримлит-приложения

#### 1. Загрузка файла для EDA и обучения моделей
Для начала работы необходимо загрузить датасет в разделе `Загрузите ZIP-файл для обучения`, рекомендуем воспользоваться следующими подготовленными опциями:
> <span style="color: red; font-weight: bold; text-decoration: underline;">ВНИМАНИЕ!</span> Образцы данных для использования приложения приведены в этом блоке:
- [полный набор данных для EDA и обучения моделей `Georgia.zip`](https://storage.yandexcloud.net/ecg-project/Georgia.zip)
- [комплект тестовых данных для инференса `test_ecg.zip`](https://storage.yandexcloud.net/ecg-project/test_ecg.zip)

![](https://storage.yandexcloud.net/ecg-project/11.gif)

#### 2. EDA
В разделе `Разведочный анализ данных` нажмите на кнопку `Провести EDA` и наслаждайтесь интерактивными графиками, посвященным описанию загруженного датасета  
![](https://storage.yandexcloud.net/ecg-project/2.gif)

#### 3. Обучение собсвтенной модели
На основе загруженного Вами датасета Вы можете обучить собсвтенную модель в разделе `Обучение модели`. Выберете один из двух видов линейных моделей, гиперпараметры и нажите `Обучить модель`  
![](https://storage.yandexcloud.net/ecg-project/3.gif)

#### 4. Предикт
Ввиду того, что сервисом пользуется только команда разработчиков и бабушка одного из них, на данном этапе названия обученных моделей формируются по принципу: `HH_MM_DD_MM_YYYY_uniqueid`  

Для формирования предикта модели Вы можете выбрать уже ранее обученную модель либо же модель, обученную Вами  
Сделать это можно в разделе `Прогноз по анализам ЭКГ`  
Для каждой выбранной модели отображаются ее вид, гиперпараметры и метрики качества  
`Загрузите ZIP-файл для прогноза диагноза` в соответствующем разделе
![](https://storage.yandexcloud.net/ecg-project/4.gif)
