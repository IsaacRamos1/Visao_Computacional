import os

# Diretório principal
base_dir = "data"

# Lista de pastas para verificar
folders = ["train", "val", "test"]
image_extensions = {".jpg"}

def count_images(directory):
    counts = {}
    if not os.path.exists(directory):
        return counts
    for subdir in os.listdir(directory):
        subdir_path = os.path.join(directory, subdir)
        if os.path.isdir(subdir_path):
            image_count = sum(
                1 for file in os.listdir(subdir_path)
                if os.path.isfile(os.path.join(subdir_path, file)) and os.path.splitext(file)[1].lower() in image_extensions
            )
            counts[subdir] = image_count
    return counts

# Imprime os resultados
for folder in folders:
    folder_path = os.path.join(base_dir, folder)
    print(f"Imagens na pasta '{folder}':")
    class_counts = count_images(folder_path)
    if class_counts:
        for class_name, count in class_counts.items():
            print(f"  Classe '{class_name}': {count} imagens")
    else:
        print("  Nenhuma classe encontrada ou pasta inexistente.")
    print()
