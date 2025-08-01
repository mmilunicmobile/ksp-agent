# Hey Gemini Code

The main entry point for this program is the main() of optimizer_gen.

Here are your tasks. Do them one at a time and check them off when you are done. Don't try testing your code and generally ignore lint errors, most are bogus.

[x] Create a method that characterizes a rocket. Take note of everything that is stored during the creation phase of a KRPCSimulatedRocket. You want to do this and record it, then activate rocket staging and recaracterize it again, and also record that. Keep repeating till the rocket has no more stages. Save the recorded data in like a JSON file.
[x] Make a method that takes one of these characterization files for a rocket, and given the fuel mass that has been used so far in the rocket, gives the proper characteristics for the rocket.
[x] Make a method that also calculates how much Delta-V a rocket would have left give a certain amount of fuel mass has been used.