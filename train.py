from data import create_midi_with_lyrics_df

train_df = create_midi_with_lyrics_df()
test_df = create_midi_with_lyrics_df(False)

import pretty_midi


def parse_midi(path):
    midi = None
    try:
        midi = pretty_midi.PrettyMIDI(path)
        midi.remove_invalid_notes()
    except Exception as e:
        raise Exception(("%s\nerror readying midi file %s" % (e, path)))
    return midi


for index, row in train_df.head(50).iterrows():
    midi = parse_midi(row['midi_path'])
    print([i.name for i in midi.instruments])
