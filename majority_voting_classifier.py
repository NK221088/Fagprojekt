def BaselineModel(TappingTest, ControlTest, TappingTrain, ControlTrain, theta):
    
    N = (len(ControlTest) + len(TappingTest))
    
    if len(TappingTrain) < len(ControlTrain):
        return len(ControlTest) / N
    else:
        return len(TappingTest) / N