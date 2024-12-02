import os
import random
import shutil

# Caminhos das pastas
base_dir = "data"
test_dir = os.path.join(base_dir, "test")
val_dir = os.path.join(base_dir, "val")

# Classes
classes = ["BASH", "BBH", "GMA", "SHC", "TSH"]

# Número de imagens a serem movidas por classe
num_to_move = 20

# Extensões comuns de imagem
image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tiff"}

# Função para mover imagens
def move_images(class_name):
    test_class_dir = os.path.join(test_dir, class_name)
    val_class_dir = os.path.join(val_dir, class_name)

    # Criar diretório da classe em 'val' se não existir
    os.makedirs(val_class_dir, exist_ok=True)

    # Listar imagens na pasta de teste
    images = [
        os.path.join(test_class_dir, file)
        for file in os.listdir(test_class_dir)
        if os.path.isfile(os.path.join(test_class_dir, file)) and os.path.splitext(file)[1].lower() in image_extensions
    ]

    # Verificar se há imagens suficientes para mover
    if len(images) < num_to_move:
        print(f"Classe '{class_name}': Imagens insuficientes para mover ({len(images)} disponíveis).")
        return

    # Selecionar imagens aleatórias para mover
    images_to_move = random.sample(images, num_to_move)

    # Mover as imagens
    for img in images_to_move:
        shutil.move(img, os.path.join(val_class_dir, os.path.basename(img)))

    print(f"Movidas {num_to_move} imagens da classe '{class_name}' de 'test' para 'val'.")

# Executar para todas as classes
for class_name in classes:
    move_images(class_name)

print("Transferência concluída.")
