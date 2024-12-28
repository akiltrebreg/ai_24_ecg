import os

def print_tree(dir_path, prefix="", limit=20):
    # Получаем и сортируем все элементы в директории
    entries = sorted(os.listdir(dir_path))
    entries_count = len(entries)

    # Отсекаем только первые 'limit' (20) элементов
    display_entries = entries[:limit]

    # Проходим по каждому элементу из сокращённого списка
    for index, entry in enumerate(display_entries):
        full_path = os.path.join(dir_path, entry)
        is_dir = os.path.isdir(full_path)

        # Определяем, последний ли это элемент в 'display_entries'
        if index == len(display_entries) - 1:
            connector = "`-- "
            new_prefix = prefix + "    "
        else:
            connector = "|-- "
            new_prefix = prefix + "|   "

        print(prefix + connector + entry)

        # Если элемент — директория, рекурсивно обходим её содержимое
        if is_dir:
            print_tree(full_path, new_prefix, limit)

    # Если всего в директории больше, чем 'limit' элементов, добавляем многоточие
    if entries_count > limit:
        print(prefix + "|-- ...")

if __name__ == "__main__":
    # Для примера используем текущую директорию
    print_tree(".")
