import copy
import torch.utils.data as data

class UniHan(data.Dataset):
    """ Dataset.
    Args:
        transform (callable, optional): A function/transform that takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    def __init__(self, filepath, sources, target):
        self.lines = [l.strip('\n').split('\t') for l in open(filepath)]
        self.sources = sources
        self.target = target

    def __add__(self, other):
        assert type(self) == type(other)
        assert all([(x == y) for x, y in zip(self.sources, other.sources)])
        assert self.target == other.target
        ret = copy.deepcopy(self)
        ret.lines += other.lines
        return ret

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (input sources, output syllable).
        """
        line = self.lines[index]

        return [line[i] for i in self.sources], line[self.target]

    def __len__(self):
        return len(self.lines)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        return fmt_str

    def cvfold(self, folds):
        ret = []
        fold_size = len(self) // folds
        for i in range(folds):
            train = copy.deepcopy(self)
            train.lines = train.lines[:i*fold_size] + train.lines[(i+1)*fold_size:]
            test = copy.deepcopy(self)
            test.lines = test.lines[i*fold_size:(i+1)*fold_size]
            ret.append((train, test))
        return ret
