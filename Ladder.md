<style>
body {
    background-color: #000;
    color: #fff;
}
</style>

# Ladder
An algorithm to evolve a master player for a board game using neural networks.

## Game
n x m rectangle\
2 players making moves in turns\
Machine player's symbol is X\
Human player symbol is O\
Either of them can be the first player

## Neural network
model=Sequential()\
model.add(Dense(n_neurons,activation='relu',input_shape=(1,)))\
model.add(Dense(n_rows*n_cols))\
The input shape is 1,meaning that every board state will be converted to one integer value.\
We take a row-continuous implementation of the board rectangle.\
This will be an n x m position integer, and at each position we place the number that identifies the symbol that is occupying that square.\
0 - empty square\
1 - square occupied by an X\
2 - square occupied by an O\
So --XO -> 5+1 (to avoid the 0 integer, we add 1)

## Training
Model fit is run after  each game