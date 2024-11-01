import cv2
import SGHIST

if __name__ == '__main__':
    dataloader = SGHIST.Dataloader(batch_size=2, size=500, shuffle=True, description=True)

    train_dataloader = dataloader.get_train_dataloader()
    # val_dataloader = dataloader.get_val_dataloader()
    # test_dataloader = dataloader.get_test_dataloader()

    EPOCHS = 10
    for epoch in range(0, EPOCHS):
        for image, label in train_dataloader:
            image = image[0].detach().cpu().numpy().transpose(1, 2, 0)
            print(f'Class label: {label[0]}')
            cv2.imshow('image', image)

            if cv2.waitKey(0) == ord('q'):
                break