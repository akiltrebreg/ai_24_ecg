Набор данных ЭКГ был получен из датасета для соревнования PhysioNet 2020 года. 
https://drive.google.com/drive/folders/1EI8XNLSLe8Hs2_HM0K_aQ5C5h2focnC5?usp=sharing
Представленные анализы принадлежат жителям штата Джорджия, США. 
Набор содержит 10 344 показания ЭКГ в 12 отведениях длиной 10 секунд с частотой дискретизации 500 Гц. 
Заключения профессиональных экспертов по данным электрокардиограммам содержат как различные сердечные заболевания, так и вердикт о хорошей деятельности сердечной мышцы (синусовый ритм). 

Результат по каждому пациенту хранится в виде двух файлов следующих форматов:
«.hea», который содержит короткий текст, описывающий физиологический сигнал
«.mat», содержащий сигнал в виде массива значений и метаданные анализа

Используемый в проекте датасет был собран из предварительной обработанных «.mat» файлов.

Датасет содержал в себе исключительно кодировки заболеваний и был дополнен расшифровкой SNOMED CT. Это один из стандартов США, предназначенный для электронного обмена клинической медицинской информацией. Каждому коду заболевания соответствует установленные полное и краткое наименования.

### Полученный датафрейм с ЭКГ пациентов содержит следующие колонки:
 1. id – уникальный идентификатор ЭКГ пациента, строковый тип
 2. one – первое значение сигнала первого отведения, числовой тип
 3. two – первое значение сигнала второго отведения, числовой тип
 4. three – первое значение сигнала третьего отведения, числовой тип
 5. aVR – первое значение сигнала четвертого отведения, числовой тип
 6. aVL – первое значение сигнала пятого отведения, числовой тип
 7. aVF – первое значение сигнала шестого отведения, числовой тип
 8. V1 – первое значение сигнала седьмого отведения, числовой тип
 9. V2 – первое значение сигнала восьмого отведения, числовой тип
 10.V3 – первое значение сигнала девятого отведения, числовой тип
 11. V4 – первое значение сигнала десятого отведения, числовой тип
 12. V5 – первое значение сигнала одиннадцатого отведения, числовой тип
 13. V6 – первое значение сигнала двенадцатого отведения, числовой тип
 14. gender – пол пациента, строковый тип
 15. age – возраст пациента, числовой тип
 16. labels – код заболевания, числовой тип
 17. signal – массив с численными значениями сигнала
 18. disease_name – наименование заболевания, строковый тип
 19. short_disease_name – аббревиатура заболевания, строковый тип

## Также были выделены характеристики сигналов для каждого отведения:

Спектральная энтропия

one_spectral_entropy – спектральная энтропия для первого отведения
two_spectral_entropy – спектральная энтропия для второго отведения
three_spectral_entropy – спектральная энтропия для третьего отведения
aVR_spectral_entropy – спектральная энтропия для четвертого отведения
aVL_spectral_entropy – спектральная энтропия для пятого отведения
aVF_spectral_entropy – спектральная энтропия для шестого отведения
V1_spectral_entropy – спектральная энтропия для седьмого отведения
V2_spectral_entropy – спектральная энтропия для восьмого отведения
V3_spectral_entropy – спектральная энтропия для девятого отведения
V4_spectral_entropy – спектральная энтропия для десятого отведения
V5_spectral_entropy – спектральная энтропия для одиннадцатого отведения
V6_spectral_entropy – спектральная энтропия для двенадцатого отведения

Спектральная дисперсия

one_spectral_variation
two_spectral_variation
three_spectral_variation
aVR_spectral_variation
aVL_spectral_variation
aVF_spectral_variation
V1_spectral_variation
V2_spectral_variation
V3_spectral_variation
V4_spectral_variation
V5_spectral_variation
V6_spectral_variation

Mel-частотные кепстральные коэффициенты

one_mfcc
two_mfcc
three_mfcc
aVR_mfcc
aVL_mfcc
aVF_mfcc
V1_mfcc
V2_mfcc
V3_mfcc
V4_mfcc
V5_mfcc
V6_mfcc

Уменьшение амплитуды

one_spectral_decrease
two_spectral_decrease
three_spectral_decrease
aVR_spectral_decrease
aVL_spectral_decrease
aVF_spectral_decrease
V1_spectral_decrease
V2_spectral_decrease
V3_spectral_decrease
V4_spectral_decrease
V5_spectral_decrease
V6_spectral_decrease

Среднее абсолютное отклонение

one_mean_abs_diff
two_mean_abs_diff
three_mean_abs_diff
aVR_mean_abs_diff
aVL_mean_abs_diff
aVF_mean_abs_diff
V1_mean_abs_diff
V2_mean_abs_diff
V3_mean_abs_diff
V4_mean_abs_diff
V5_mean_abs_diff
V6_mean_abs_diff

Среднее значение разностей 

one_mean_diff
two_mean_diff
three_mean_diff
aVR_mean_diff
aVL_mean_diff
aVF_mean_diff
V1_mean_diff
V2_mean_diff
V3_mean_diff
V4_mean_diff
V5_mean_diff
V6_mean_diff

Абсолютная энергия

one_abs_energy
two_abs_energy
three_abs_energy
aVR_abs_energy
aVL_abs_energy
aVF_abs_energy
V1_abs_energy
V2_abs_energy
V3_abs_energy
V4_abs_energy
V5_abs_energy
V6_abs_energy

Энтропия

one_enthropy
two_enthropy
three_enthropy
aVR_enthropy
aVL_enthropy
aVF_enthropy
V1_enthropy
V2_enthropy
V3_enthropy
V4_enthropy
V5_enthropy
V6_enthropy

Коэффициент асимметрии

one_skewness
two_skewness
three_skewness
aVR_skewness
aVL_skewness
aVF_skewness
V1_skewness
V2_skewness
V3_skewness
V4_skewness
V5_skewness
V6_skewness

 Коэффициент эксцесса/островершинности
 
one_kurtosis
two_kurtosis
three_kurtosis
aVR_kurtosis
aVL_kurtosis
aVF_kurtosis
V1_kurtosis
V2_kurtosis
V3_kurtosis
V4_kurtosis
V5_kurtosis
V6_kurtosis


Всего в заключениях по результат электрокардиограмм содержится 67 заболеваний.
При изучении метаданных было обнаружено 150 пропущенных значений возраста (колонка age). Пропуски были заполненными медианными значениями возраста отдельно для мужчин и отдельно для женщин.
