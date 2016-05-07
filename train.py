from domain_adaptation import test_lr

test_lr(train=['yelp','twitter'], n_train=[500000,20000],
        feature_extractor='weighted')
test_lr(train=['yelp','twitter'], n_train=[500000,100000],
        feature_extractor='weighted')
