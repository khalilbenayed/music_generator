from mido import MidiFile, Message, MidiTrack, MetaMessage
import numpy as np

lowerBound = 24
upperBound = 102
span = upperBound - lowerBound


def midiToNoteStateMatrix(midi_file, squash=True, span=span):
    mid = MidiFile(midi_file)
    pattern = mid.tracks

    timeleft = [0 for track in pattern]
    posns = [0 for track in pattern]

    statematrix = []
    time = 0

    state = [[0, 0] for x in range(span)]
    statematrix.append(state)
    condition = True
    while condition:
        if time % (mid.ticks_per_beat / 4) == (mid.ticks_per_beat / 8):
            # Crossed a note boundary. Create a new state, defaulting to holding notes
            oldstate = state
            state = [[oldstate[x][0], 0] for x in range(span)]
            statematrix.append(state)
        for i in range(len(timeleft)):  # For each track
            if not condition:
                break
            while timeleft[i] == 0:
                track = pattern[i]
                pos = posns[i]
                evt = track[pos]

                if not evt.is_meta:
                    if evt.type == 'note_off' or evt.type == 'note_on':
                        if (evt.note < lowerBound) or (evt.note >= upperBound):
                            pass
                        else:
                            if evt.type == 'note_off' or evt.velocity == 0:
                                state[evt.note - lowerBound] = [0, 0]
                            else:
                                state[evt.note - lowerBound] = [1, 1]
                elif evt.is_meta:
                    if evt.type == 'time_signature':
                        if evt.numerator not in (2, 4):
                            # We don't want to worry about non-4 time signatures
                            out = statematrix
                            condition = False
                            break
                    else:
                        pass
                try:
                    if not track[pos + 1].is_meta:
                        timeleft[i] = track[pos+1].time
                    posns[i] += 1
                except IndexError:
                    timeleft[i] = None

            if timeleft[i] is not None:
                timeleft[i] -= 1

        if all(t is None for t in timeleft):
            break

        time += 1

    S = np.array(statematrix)
    statematrix = np.hstack((S[:, :, 0], S[:, :, 1]))
    statematrix = np.asarray(statematrix).tolist()
    return statematrix


def noteStateMatrixToMidi(state_matrix, name="example", span=span):
    state_matrix = np.array(state_matrix)
    if not len(state_matrix.shape) == 3:
        state_matrix = np.dstack((state_matrix[:, :span], state_matrix[:, span:]))
    state_matrix = np.asarray(state_matrix)
    pattern = MidiFile()

    pattern.add_track()
    track = pattern.tracks[0]

    span = upperBound - lowerBound
    tickscale = 55

    lastcmdtime = 0
    prevstate = [[0, 0] for x in range(span)]
    for time, state in enumerate(state_matrix + [prevstate[:]]):
        offNotes = []
        onNotes = []
        for i in range(span):
            n = state[i]
            p = prevstate[i]
            if p[0] == 1:
                if n[0] == 0:
                    offNotes.append(i)
                elif n[1] == 1:
                    offNotes.append(i)
                    onNotes.append(i)
            elif n[0] == 1:
                onNotes.append(i)
        for note in offNotes:
            track.append(Message(type='note_off', time=(time - lastcmdtime) * tickscale, note=note + lowerBound))
            lastcmdtime = time
        for note in onNotes:
            track.append(Message(type='note_on', time=(time - lastcmdtime) * tickscale, velocity=40, note=note + lowerBound))
            lastcmdtime = time

        prevstate = state

    eot = MetaMessage('end_of_track')
    track.append(eot)

    pattern.save(f'{name}.mid')

