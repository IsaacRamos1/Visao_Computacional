import os
import random

# Caminho para a pasta da classe 'BASH'
bash_dir = "data/test/TSH"

# Número de imagens desejado
target_count = 80

# Extensões comuns de imagem
image_extensions = {".jpg"}

# Listar todas as imagens na pasta
images = [
    os.path.join(bash_dir, file)
    for file in os.listdir(bash_dir)
    if os.path.isfile(os.path.join(bash_dir, file)) and os.path.splitext(file)[1].lower() in image_extensions
]

# Verificar se a redução é necessária
current_count = len(images)
if current_count <= target_count:
    print(f"A pasta já possui {current_count} imagens, não é necessário reduzir.")
else:
    # Selecionar aleatoriamente imagens para manter
    images_to_keep = set(random.sample(images, target_count))
    
    # Excluir as imagens excedentes
    images_to_remove = [img for img in images if img not in images_to_keep]
    for img in images_to_remove:
        os.remove(img)
    
    print(f"Reduzido de {current_count} para {target_count} imagens na pasta 'BASH'.")
