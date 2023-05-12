from models.time2vec import *
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default="images")
    parser.add_argument("--model")
    parser.add_argument("--save", default="time2vec")
    args = parser.parse_args()

    transforms = Compose([
        Resize((224, 224)),
        ToTensor(),
        Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    dataset = ImageFolder(
        args.path,
        transforms,
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
    
    model = Time2Vec.load_from_checkpoint(args.model, map_location="cpu")

    vectors, labels = dataloader2vec(dataloader, model, return_label=True)

    np.savez(args.save, vectors, labels)


    