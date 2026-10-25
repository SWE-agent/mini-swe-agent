class Anthropic:
    def __init__(self, *args, **kwargs):
        self.messages = self

    def create(self, *args, **kwargs):
        raise NotImplementedError
