from tqdm import tqdm
from datasets import ForexDataWithWindow
import models.time2vec as time2vec
import pandas as pd

def get_vectors(dataset, model=None):
    for i in tqdm(range(0, len(dataset))):
        data = dataset.getInterval(i)
        x = data.iloc[:30]
        if model is None:
            vec = time2vec.time2vec(x, checkpoint_path="epoch=93-step=7896.ckpt")
        else:
            vec = time2vec.time2vec(x, model)
        
        yield x.iloc[0].name, vec

def save_vectors(dataset, model, filename="vecs.csv", batch=100):
    vecs = []
    for i, vec in enumerate(get_vectors(dataset, model)):
        vecs.append(vec)
        if i % batch == 0:
            vecs_df = pd.DataFrame(vecs, columns=["Time", "vec"])
            vecs_df.to_csv(filename, mode="a", header=False)
            vecs = []

if __name__ == "__main__":
    dataset = ForexDataWithWindow("./data/USDJPY_H1.csv", header=0, normalize=False, data_order="tohlc", input_duration=31, time_index=True)
    model = time2vec.Time2Vec.load_from_checkpoint("time2vec_usdjpy.ckpt", map_location="cpu")
    model.eval()
    save_vectors(dataset, model)