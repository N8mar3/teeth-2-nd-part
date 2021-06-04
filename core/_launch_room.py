if __name__ == '__main__':
    import json
    from _main_distribution_room import train_model
    train_model(*list(json.load(open('training__', 'r')).values()))
