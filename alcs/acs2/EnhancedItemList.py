
class EnhancedItemList(set):

    # def __init__(self):
    #     set.__init__(self, [set() for _ in range(self.cfg.classifier_length)])

    def add_item(self, item, prob):
        self.add()
