from dataset.forest_dataset import ForestCaptionDataset

if __name__ == '__main__':

    dataset = ForestCaptionDataset(
    im_path=r"../data/image_text"
    )

    img, cond = dataset[0]
    print(img.shape)   
    print(cond["text"])     
