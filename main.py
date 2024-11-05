import cv2
import SGHIST

if __name__ == '__main__':
    print('Executando...')
    batch_size = 2
    dataloader = SGHIST.Dataloader(batch_size=batch_size, size=500, shuffle=True, description=True)

    train_dataloader = dataloader.get_train_dataloader()
    val_dataloader = dataloader.get_val_dataloader()
    test_dataloader = dataloader.get_test_dataloader()

    
    for image, label in train_dataloader:
        for batch in range(batch_size):
            image = image.detach().cpu().numpy()[batch].transpose(1, 2, 0)
            print(f'Class label: {label[batch]}')
            cv2.imshow('image', image)

            if cv2.waitKey(0) == ord('q'):
                break