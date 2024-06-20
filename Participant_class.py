from seed import set_seeds
set_seeds()
class individual_participant_class:
    def __init__(self, name):
        self.name = name
        self.events = {}
        self.raw_intensity = None
        self.raw_od = None
        self.raw_haemo_unfiltered = None
        self.raw_haemo = None