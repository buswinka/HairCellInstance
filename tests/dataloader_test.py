import src.dataloader

def test_dataloader_function():

    path = '/media/DataStorage/Dropbox (Partners HealthCare)/HairCellInstance/data/test'
    data = src.dataloader.dataset(path)

    image, mask = data[0]

    assert len(data) == 1

