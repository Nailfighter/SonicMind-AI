import cmd
import threading
from LiveAudioFX import (
    LiveAudioProcessor,
    GainEffect,
    DistortionEffect,
    DelayEffect,
    ReverbEffect,
    CompressorEffect,
    EQEffect
)

class PedalBoardShell(cmd.Cmd):
    intro = 'Welcome to the Pedal Board. Type help or ? to list commands.\n'
    prompt = '(pedal-board) '

    def __init__(self, processor):
        super().__init__()
        self.processor = processor

    def do_list(self, arg):
        'List active effects: list'
        self.processor.list_effects()

    def do_add(self, arg):
        'Add an effect: add [effect_name]'
        try:
            effect_name, *params = arg.split()
            effect = self._create_effect(effect_name, params)
            if effect:
                self.processor.add_effect(effect)
                print(f"Added {effect_name}.")
        except Exception as e:
            print(f"Error adding effect: {e}")

    def do_remove(self, arg):
        'Remove an effect: remove [index]'
        try:
            index = int(arg)
            self.processor.remove_effect(index)
            print(f"Removed effect at index {index}.")
        except (ValueError, IndexError) as e:
            print(f"Error removing effect: {e}")

    def do_set(self, arg):
        'Set a parameter for an effect: set [effect_index] [param_name] [value]'
        try:
            index, param, value = arg.split()
            index = int(index)
            value = float(value)
            effect = self.processor.effects[index]
            effect.set_parameter(param, value)
            print(f"Set {param} to {value} for effect {index}.")
        except (ValueError, IndexError) as e:
            print(f"Error setting parameter: {e}")

    def do_quit(self, arg):
        'Exit the pedal board: quit'
        return True

    def _create_effect(self, name, params):
        if name == 'gain':
            gain = float(params[0]) if params else 1.0
            return GainEffect(gain=gain)
        elif name == 'distortion':
            drive = float(params[0]) if params else 0.5
            return DistortionEffect(drive=drive)
        elif name == 'delay':
            delay_time = float(params[0]) if params else 0.5
            feedback = float(params[1]) if len(params) > 1 else 0.5
            mix = float(params[2]) if len(params) > 2 else 0.5
            return DelayEffect(delay_time=delay_time, feedback=feedback, mix=mix)
        elif name == 'reverb':
            room_size = float(params[0]) if params else 0.5
            decay = float(params[1]) if len(params) > 1 else 0.5
            damping = float(params[2]) if len(params) > 2 else 0.5
            mix = float(params[3]) if len(params) > 3 else 0.5
            return ReverbEffect(room_size=room_size, decay=decay, damping=damping, mix=mix)
        elif name == 'compressor':
            threshold = float(params[0]) if params else -20.0
            ratio = float(params[1]) if len(params) > 1 else 4.0
            attack = float(params[2]) if len(params) > 2 else 0.01
            release = float(params[3]) if len(params) > 3 else 0.1
            return CompressorEffect(threshold=threshold, ratio=ratio, attack=attack, release=release)
        elif name == 'eq':
            return EQEffect()
        else:
            print(f"Unknown effect: {name}")
            return None

def main():
    processor = LiveAudioProcessor()
    processor.start()

    shell = PedalBoardShell(processor)
    shell.cmdloop()

    processor.stop()

if __name__ == '__main__':
    main()