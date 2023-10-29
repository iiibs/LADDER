from keras.models import Sequential
from keras.models import load_model
from keras.layers import Dense
from keras.layers import Activation
from keras.callbacks import Callback
import random
import numpy as np
import copy
import matplotlib.pyplot as plt
import pickle
import os
import shutil
import time
import threading
import math
import psutil

# Global constants
n_states=3 # because there are 3 states of a square: 1. empty, 2. X, 3. O
n_rows=1
n_cols=4
n_connect=2
n_neurons=n_states**(n_rows*n_cols) # number of neurons in a hidden layer
n_max_candidates=n_neurons #10 #40 # maximum number of candidates that will be tried
n_gamepairs=n_states*n_neurons #10 # The number of game pairs that will make up a match between the candidate and a member
n_epochs=32 #2 #1000 #2000 #1024 #65535 # Number of epochs performed to train after one move (standard=32)
next_rank_rate=(0.5+1)/2 #(0.50+(0.5+1)/2)/2 # This is the rate that means that the winner should be in a higher league
n_max_members=n_rows*n_cols # This many members can be at maximum on a certain rank of the ladder
b_repeatable_random=False #True # when this is false, then we can get new results
ladder_path=f"C:\\DESIRE_DATA\\CONNECT{n_connect}"
verbose=0 #'learn' #'details' #False #'spec' #True
b_plot_game=False #True
b_plot_match=False #True

def print_board(board):
 for row in board:
  for col in row:
   print(col,end='')
  print()
 print()

def board_to_int(board):
 """
 Convert a 2D game board to an integer representation.
 """
 int_value=0
 for row in board:
  for cell in row:
   int_value = int_value * n_states  # Shift left by one digit
   if cell == 'X':
    int_value += 1
   elif cell == 'O':
    int_value += 2
 return int_value+1
   
def int_to_board(int_value):
 """
 Convert an integer representation to a 2D game board.
 """
 int_value-=1
 board = [['-' for _ in range(n_cols)] for _ in range(n_rows)]
 for i in range(n_rows-1, -1, -1):
  for j in range(n_cols-1, -1, -1):
   remainder = int_value % n_states
   if remainder == 1:
    board[i][j] = 'X'
   elif remainder == 2:
    board[i][j] = 'O'
   int_value //= n_states  # Shift right by one digit
 return board

def check_dir(board, row, col, dx, dy, symbol):
 for i_connect in range(1, n_connect):
  new_row = row + i_connect * dy
  new_col = col + i_connect * dx
  if new_row < 0 or new_row >= len(board) \
   or new_col < 0 or new_col >= len(board[0]) \
   or board[new_row][new_col] != symbol:
   return False
 return True

def is_winner(board, symbol):
 # Define all possible directions to check
 directions=[
  ( 0, -1), # north
  ( 1, -1), # northeast
  ( 1,  0), # east
  ( 1,  1), # southeast
  ( 0,  1), # south
  (-1,  1), # southwest
  (-1,  0), # west
  (-1, -1), # northwest
  ]  
 for row in range(len(board)):
  for col in range(len(board[0])):
   if board[row][col]==symbol:
    for dx, dy in directions:
     if check_dir(board, row, col, dx, dy, symbol):
      return True
 return False

def has_game_ended_result(board):
 if is_winner(board,'X'):
  return True,1.0
 if is_winner(board,'O'):
  return True,0.0
 if all(cell != '-' for row in board for cell in row):
  return True,0.5
 return False,-1

def move_index_to_coordinates(index):
 row=index // n_cols  # Integer division to get the row
 col=index % n_cols   # Modulus operation to get the column
 return (row,col)

def move_coordinates_to_index(move):
 return move[0]*n_cols+move[1]

def select_move(board,predicted_probabilities):
 # Set the probabilities of the non-empty cells to zero
 for i_row in range(n_rows):
  for i_col in range(n_cols):
   if board[i_row][i_col]!='-':
    move_index=move_coordinates_to_index((i_row,i_col))
    predicted_probabilities[move_index]=0.0 # Not legal move
 # Make sure that probabilities add up to 1 even after possibly zeroing out some
 sum_probabilities=0
 for i_probability in range(len(predicted_probabilities)):
  sum_probabilities+=predicted_probabilities[i_probability]
 # If there is no good move, then give up
 if sum_probabilities==0:
  return -1
 for i_probability in range(len(predicted_probabilities)):
  predicted_probabilities[i_probability]/=sum_probabilities
 # Generate a random move index based on the probabilities
 move_index=np.random.choice(len(predicted_probabilities),p=predicted_probabilities)
 # Get the corresponding (row, col) coordinates of the selected move
 selected_move=move_index_to_coordinates(move_index)
 return selected_move # Return selected move

def softmax(x):
 e_x = np.exp(x - np.max(x))  # Subtracting max(x) for numerical stability
 return e_x / e_x.sum(axis=0)

def update_board(board,move,current_player):
 row=move[0]
 col=move[1]
 board[row][col]=current_player
 return board

# Custom callback to print progress every nth epoch
#  and to store the loss values of the test data at the same time
class PrintProgressStoreTestLosses(Callback):
 def __init__(self,model,n=1,x_test=None,y_test=None):
  super().__init__()
  self.n = n
  self.x_test=x_test
  self.y_test=y_test
  #self.test_losses=[]
  self.test_mses=[]
  self.model=model
  self.epoch_pred_test_values=[]
 def on_epoch_end(self,epoch,logs=None):
  if epoch % self.n == 0:
   pred_test_values=self.model.predict(np.array([board_to_int(['-','-','-','-',])]),verbose=0)[0]
   formatted_pred_test_values=['{:.8f}'.format(value) for value in pred_test_values]
   if verbose=='details':
    print("learn:",formatted_pred_test_values)
   self.epoch_pred_test_values.append(pred_test_values)
   #print(f"Epoch {epoch + 1}/{self.params['epochs']}")
   #print(logs)
   #print(f" - train loss: {logs['loss']:.4f} - train mse: {logs['mse']:.4f}")
   #print(f" - train loss: {logs['loss']:.4f}")
   #print(f" - train mse: {logs['mse']:.4f}")
   #if self.x_test and self.y_test:
   # results=self.model.evaluate(np.array(self.x_test),np.array(self.y_test),verbose=0)
   # self.test_losses.append(results[0])
   # self.test_mses.append(results[1])
   # print(f' - test loss: {results[0]:.4f} - test mse: {results[1]:.4f}')
   # print(f' - test loss: {results[0]:.4f}')
   # print(f' - test mse: {results[1]:.4f}')

def run_game(i_gamepair,candidate,member,first,gt_plot_values,pred_plot_values):
 board=init_board()
 if "candidate" in first:
  current_player='X' # Game starts with the candidate player 'X' move
 else:
  current_player='O' # Game starts with the member player 'O' move
 x_train=[] # List to store board states (inputs for nn)
 y_train=[] # List to store game results (targets for nn)
 boards=[] # Store board states for the current game where X is to move
 predictions=[] # Store predictions
 moves=[] # Store moves
 game_ended=False
 while not game_ended:
  if current_player=='X':
   # It's the candidate's move, so get a probability distribution from the candidate's model
   if verbose=='memory':
    import keras.backend as K
    prev_mem=get_mem()
   predicted_values=candidate.predict(np.array([board_to_int(board)]),verbose=0)[0]
   if verbose=='memory':
    print_mem_delta(prev_mem,"candidate.predict")
   # Select a candidate move based on the probabilities
   move=select_move(board,predicted_values)
   if move==-1:
    # The candidate nn found no good move, it gives up (all probabilities were zero)
    game_ended=True
    game_result=0.0
    if verbose=='learn':
     print("--- candidate gave up ---")
    break
   # Store the current board state and the selected move
   boards.append(board_to_int(board))
   predictions.append(predicted_values)
   moves.append(move_coordinates_to_index(move))
   if verbose=='details':
    print(board,' -> ',board_to_int(board))
    predicted_values_formatted=['{:.8f}'.format(value) for value in predicted_values]
    print(predicted_values_formatted)
    print(move)
  else:
   # It's the member's move, so get a probability distribution from the member's model
   if verbose=='memory':
    prev_mem=get_mem()
   predicted_values=member.predict(np.array([board_to_int(board)]),verbose=0)[0]
   if verbose=='memory':
    print_mem_delta(prev_mem,"member.predict")
   # Select a member move based on the probabilities
   move=select_move(board,predicted_values)
   if move==-1:
    # The member nn found no good move, it gives up (all probabilities were zero)
    game_ended=True
    game_result=1.0
    if verbose=='learn':
     print("--- member gave up ---")
    break
  # Update the board with the chosen move
  update_board(board,move,current_player)
  if verbose=='learn':
   print_board(board)
  # Check game result if the game has ended
  game_ended,game_result=has_game_ended_result(board)
  if game_ended:
   break
  # Switch to the other player for the next move
  current_player='X' if current_player=='O' else 'O'
 if verbose=='learn':
  print ("   --- game over ",game_result," ---")
 target_value=game_result
 # Append all board states for this game to x_train
 x_train.extend(boards)
 for x_train_index in range(len(x_train)):
  # Set prediction vector element to game result as target value where it was the selected move
  predictions[x_train_index][moves[x_train_index]]=target_value
 y_train.extend(predictions)
 # Train the candidate using x_train and y_train data
 if verbose=='learn':
  print ("   --- train ---")
 if verbose=='details':
  print("==========")
  for _ in range(len(x_train)):
   print_board(int_to_board(x_train[_]))
   y_train_formatted=['{:.8f}'.format(value) for value in y_train[_]]
   print(y_train_formatted)
   print("----------")
  print("==========")
 # Prepare test data and the callback
 x_test=[]
 y_test=[]
 #x_test.append(board_to_int(['-', 'X', '-', '-'])); y_test.append(1.0)
 #x_test.append(board_to_int(['-', '-', 'X', '-'])); y_test.append(1.0)
 #x_test.append(board_to_int(['X', '-', '-', '-'])); y_test.append(0.0)
 #x_test.append(board_to_int(['-', '-', '-', 'X'])); y_test.append(0.0)
 fit_callback=PrintProgressStoreTestLosses(candidate,n=1,x_test=x_test,y_test=y_test)

 #for x, y in zip(x_train, y_train):
 # if int_to_board(x)==[['-','-','-','-']]:
 #  print("input: ",int_to_board(x), end=' -> ')
 #  targets_formatted=[f'{:.8f}'format(value) for value in y]
 #  print("target: ",', '.join(targets_formatted))

 history=candidate.fit(
  np.array(x_train),
  np.array(y_train),
  epochs=n_epochs,
  batch_size=32,
  #validation_split=0.2,
  verbose=0,
  callbacks=[fit_callback]
  )

 if b_plot_game:
  fig=plt.figure()
  fig.canvas.manager.window.wm_geometry("+500+100") #+horiz+vert -> upper left corner
  learn_predictions=fit_callback.epoch_pred_test_values
  elements = [[] for _ in range(4)]
  for epoch_predictions in learn_predictions:
   for i, value in enumerate(epoch_predictions):
    elements[i].append(value)
  for i in range(4):
   plt.plot(elements[i], label=f'Element {i}')
  plt.title(f'Game #{i_gamepair}, {first}')
  plt.xlabel('Epoch')
  plt.ylabel('Probability')
  plt.style.use('dark_background')
  plt.legend()
  plt.savefig(f'C:\\BAREK\\DESIRE_DATA\\CONNECT2\\figures\\Figure_{i_gamepair}_{first}.png')
  plt.show()
 
 #for x, y in zip(x_train, y_train):
 # if int_to_board(x)==[['-','-','-','-']]:
 #  print("input: ",int_to_board(x), end=' -> ')
 #  targets_formatted=[f'{:.8f}'.format(value) for value in y]
 #  print("target: ",', '.join(targets_formatted))

 gt_plot_value=[0.5,1.0,1.0,0.5] # for ---- board 0.5 1.0 1.0 0.5 is the ground truth
 gt_plot_values.append(gt_plot_value)
 pred_plot_value=candidate.predict(np.array([1]),verbose=0)[0]
 pred_plot_values.append(pred_plot_value)

 #train_loss = history.history['loss']
 #validation_loss = history.history['val_loss']
 #test_loss = fit_callback.test_losses
 #test_loss = fit_callback.test_mses
 #actual=y_test[0]
 #predicted=test_mses[0]
 #squared_difference=(actual-predicted)**2
 #plot_history(train_loss,validation_loss,test_loss)
 #plot_history(train_loss,validation_loss)
 #print_spec_test_results(candidate)

 return game_result

def play_game(machine,first):
 board=init_board()
 print_board(board)
 if "machine" in first:
  current_player='X' # Game starts with the machine player 'X' move
 else:
  current_player='O' # Game starts with the human player 'O' move
 game_ended=False
 while game_ended!=True:
  if current_player=='X':
   # It's the machine's move, so get a probability distribution from the machine's model
   predicted_values=machine.predict(np.array([board_to_int(board)]),verbose=0)[0]
   predicted_values_formatted=['{:.8f}'.format(value) for value in predicted_values]
   print(predicted_values_formatted)
   # Select a move based on the probabilities
   move=select_move(board,predicted_values)
   if move==-1:
    # The machine nn found no good move, it gives up (all probabilities were zero)
    game_ended=True
    game_result=0.0
    print("--- machine gave up ---")
    break
   # Update the board with the chosen move
   update_board(board,move,current_player)
  else:
   # It's the humans's move, so get one
   make_human_move(board)
  print_board(board)
  # Check if the game has ended
  game_ended,game_result=has_game_ended_result(board)
  if game_ended:
   break
  # Switch to the other player for the next move
  current_player='X' if current_player=='O' else 'O'
 return game_result

#def plot_history(train_loss, validation_loss, test_loss):
def plot_history(train_loss, validation_loss):
 plt.title('Model loss during training')
 plt.xlabel('Epoch')
 plt.ylabel('Loss')
 plt.plot(train_loss,label='train')
 plt.plot(validation_loss,label='validation')
 #plt.plot(test_loss,label='test')
 #plt.legend(['Train data','Validation data','Test data'],loc='upper right')
 plt.legend(['Train data','Validation data'],loc='upper right')
 plt.style.use('dark_background')
 plt.show()

def run_match(candidate,ladder,rank,i_member,result_candidate,result_member):
 file_name=f"{ladder_path}\\models\\model_{ladder[rank][i_member]}"
 member=load_model(file_name)
 print(f"model loaded: {file_name}")
 groundtruths=[]
 predictions=[]
 for i_gamepair in range(n_gamepairs):
  # Game where candidate is the first player
  if verbose=='memory':
   prev_mem=get_mem()
  game_result=run_game(i_gamepair,candidate,member,"candidate first",groundtruths,predictions)
  if verbose=='memory':
   print_mem_delta(prev_mem,"run_game(candidate,member)")
  if verbose=='learn':
   print(game_result)
  result_candidate+=game_result
  result_member+=(1.0-game_result)
  # Game where member is the first player
  if verbose=='memory':
   prev_mem=get_mem()
  game_result=run_game(i_gamepair,member,candidate,"member first",groundtruths,predictions)
  if verbose=='memory':
   print_mem_delta(prev_mem,"run_game(member,candidate)")
  if verbose=='learn':
   print(game_result)
  result_member+=game_result
  result_candidate+=(1.0-game_result)
 if b_plot_match:
  #plot_match_train(np.array(groundtruths),predictions)
  plt.title('Model convergence by square')
  plt.xlabel('Epochs')
  plt.ylabel('Diffs')
  diff_array=abs(gt_array-pred_array)
  diff0=diff_array[:,0]
  diff1=diff_array[:,1]
  diff2=diff_array[:,2]
  diff3=diff_array[:,3]
  plt.plot(diff0,label='diff0')
  plt.plot(diff1,label='diff1')
  plt.plot(diff2,label='diff2')
  plt.plot(diff3,label='diff3')
  plt.legend(['Diff0','Diff1','Diff2','Diff3'],loc='upper right')
  plt.show()
 return result_candidate,result_member

def print_ladder(ladder):
 for rank,members in enumerate(ladder):
  print(f' Rank #{rank}')
  for member in members:
   print(f"  {member}")

def create_ladder():
 ladder_name=f"{ladder_path}\\ladder.pkl"
 # Delete previous ladder
 try:
  os.remove(ladder_name)
 except FileNotFoundError:
  print("ladder file not found")
 # Delete previous models
 try:
  shutil.rmtree(f"{ladder_path}\\models")
 except FileNotFoundError:
  print("models folder not found")
 # Create an empty ladder
 ladder=[]
 # Create a random model
 model=create_model()
 unique_id=0
 # Save the model to a file
 #unique_str=uuid.uuid4()
 unique_str=f"{unique_id:04d}"
 print(f"basic member: {unique_str}")
 model.save(f"{ladder_path}\\models\\model_{unique_str}")
 # Put the model name on the lowest rank of the ladder
 ladder=[[str(unique_str)],]
 with open(ladder_name,'wb') as file:
   pickle.dump(ladder,file)
 return ladder,unique_id

def load_ladder():
 ladder=None
 file_name=f"{ladder_path}\\ladder.pkl"
 if os.path.exists(file_name):
  # The file exists
  with open(file_name,'rb') as file:
   ladder=pickle.load(file)
 return ladder

def create_model():
 # Build the model
 model=Sequential()
 # 
 model.add(Dense(n_neurons,activation='relu',input_shape=(1,)))
 model.add(Dense(n_rows*n_cols))
 model.add(Activation('softmax'))
 # Compile the model
 model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
 return model

def rank_empty(ladder,rank):
 if rank>=len(ladder):
  return True
 else:
  return False

def test():
 # Test the first move predictions of newly created models
 for _ in range(2):
  model=Sequential()
  model.add(Dense(3*9,activation='relu',input_shape=(1,)))
  model.add(Dense(9))
  model.compile(optimizer='adam',loss='mse',metrics=['mse'])
  # Print initial weight values
  #for layer in model.layers:
  # print(layer.get_weights())
  input=[]
  board_int=1
  board=int_to_board(board_int)
  print(board)
  input.append(board_int)
  predicted_values=model.predict(np.array(input))[0]
  print(predicted_values)

def init_board():
 board = [['-' for _ in range(n_cols)] for _ in range(n_rows)]
 return board

def put_candidate_on_ladder(candidate,unique_id,ladder,rank):
 # Save the candidate model to a file
 #unique_id=uuid.uuid4()
 unique_str=f"{unique_id:04d}"
 model_file_name=f"{ladder_path}\\models\\model_{unique_str}"
 candidate.save(model_file_name)
 # Put the candidate model unique id given rank of the ladder as the given member
 ladder_file_name=f"{ladder_path}\\ladder.pkl"
 # Check if there is free space on this rank
 if len(ladder[rank])<n_max_members:
  print(f"candidate[{unique_id:04d}] is added to rank {rank} members")
  ladder[rank].append(unique_str)
 else:
  random_int=random.choice(range(n_max_members))
  print(f"random member index: {random_int}")
  print(f"candidate[{unique_id:04d}] is replacing member[{ladder[rank][random_int]}]")
  ladder[rank][random_int]=unique_str
 with open(ladder_file_name,'wb') as file:
  pickle.dump(ladder,file)
 print("ladder and models saved")
 return ladder

def wait_for_user_input(timeout_seconds):
 time.sleep(timeout_seconds)

def make_human_move(board):
 while True:
  n=input(f"Give a number between 0 and {n_rows*n_cols-1} for a move: ")
  # Check if n is within the valid range
  if not n.isdigit():
      print("Invalid input. Please enter a valid number.")
  else:
   n_num=int(n)
  if n_num < 0 or n_num > n_rows*n_cols-1:
   print(f"Invalid move. Please choose a number between 0 and {n_rows*n_cols-1}.")
  else:
   # Calculate the position for X
   row,col=move_index_to_coordinates(n_num)
   # Check if the calculated position is still empty
   if board[row][col]=='-':
    # Place given stone at the calculated position
    board[row][col] = 'O'
    return board
   else:
    print(f"Row: {row},{col} is already occupied. Please choose a different move.")
 return

def sigmoid(x):
 return 1/(1+np.exp(-x))

def print_spec_test_results(model):
 print("\nspecific test results - begin")
 predictions=model.predict(np.array([board_to_int(['-','-','-','-',])]))[0]
 print("predictions: ",predictions)
 #sigmoidizeds=[sigmoid(x*6) for x in predictions] # at 6 sigmoid is practically zero or one
 #print("sigmoidized predictions: ",sigmoidizeds)
 probabilities=softmax(predictions)
 print("probabilities: ",probabilities)
 #probabilities=normalizer(predictions)
 #print("probabilities by normalizer: ",probabilities)
 print("\nspecific test results - end\n")

def get_mem():
 # Get available CPU memory
 available_memory = psutil.virtual_memory().available / (1024 ** 3)  # in gigabytes
 return available_memory
def print_memory():
 # Get available CPU memory
 available_memory = psutil.virtual_memory().available / (1024 ** 3)  # in gigabytes
 print(f"Available CPU memory: {available_memory:.2f} GB")
 print()
def print_mem_delta(prev_mem,str):
 # Get current CPU memory
 curr_mem = psutil.virtual_memory().available / (1024 ** 3)  # in gigabytes
 delta_mem=curr_mem-prev_mem
 if delta_mem>0.1:
  print(f"After {str} CPU memory changed by: {delta_mem:.2f} GB")
 if delta_mem<-0.1:
  print(f"After {str} CPU memory changed by: {delta_mem:.2f} GB")

def train_ladder(ladder,unique_id):
 print_ladder(ladder)
 loop_level="next candidate"
 while True:
  if loop_level=="next candidate":
   if unique_id==n_max_candidates:
    exit(0)
   # Create new candidate
   unique_id+=1
   if verbose=='memory':
    prev_mem=get_mem()
   candidate=create_model()
   print(f"new candidate: {unique_id:04d}")
   if verbose=='memory':
    print_mem_delta(prev_mem,"candidate=create_model()")
   #if verbose=='spec':
   # print_spec_test_results(candidate)
   # Set the current rank to rank zero.
   rank=0
   print(f"candidate[{unique_id:04d}] starts on rank: {rank}")
   i_member=0
   print(f"member index: {i_member}")
  if loop_level=="next rank":
   rank+=1
   i_member=0
   # Check if rank has any members
   if rank>=len(ladder):
    # rank is empty, so create it
    ladder.append([])
    # and put candidate here
    put_candidate_on_ladder(candidate,unique_id,ladder,rank)
    print_ladder(ladder)
    loop_level="next candidate"
    continue
  # Start with the first member of the current rank
   i_member=0
   all_members_defeated_by_far=True
  if loop_level=="next member":
   i_member+=1
  # Perform a match between the current candidate and the current member on the current rank
  result_candidate=0
  result_member=0
  if verbose=='memory':
   prev_mem=get_mem()
  result_candidate,result_member=run_match(candidate,ladder,rank,i_member,result_candidate,result_member)
  print(f"candidate[{unique_id:04d}]: {result_candidate} / member[{ladder[rank][i_member]}]: {result_member}")
  if verbose=='memory':
   print_mem_delta(prev_mem,"candidate.predict")
  candidate_rate=result_candidate/(result_candidate+result_member)
  print(f"candidate[{unique_id:04d}] rate against member[{ladder[rank][i_member]}]: {candidate_rate}, and the required rate is: {next_rank_rate}")
  #if verbose=='spec':
  # print_spec_test_results(candidate)
  if not candidate_rate>=next_rank_rate:
   # The current candidate could not defeat the current member of the current rank by far
   print(f"candidate[{unique_id:04d}] could not defeat member[{ladder[rank][i_member]}] at a far rate")
   print(f"candidate[{unique_id:04d}] will stay on rank: {rank}")
   # Candidate will not go up, rather it will be added or will replace a random member of this rank
   if verbose=='memory':
    prev_mem=get_mem()
   put_candidate_on_ladder(candidate,unique_id,ladder,rank)
   print_ladder(ladder)
   if verbose=='memory':
    print_mem_delta(prev_mem,"put_candidate_on_ladder")
   loop_level="next candidate"
   continue
  if i_member+1>=len(ladder[rank]):
   print(f"candidate[{unique_id:04d}] defeated all members of the current rank at a far rate, so it goes for the next rank")
   loop_level="next rank"
   continue
  print(f"candidate[{unique_id:04d}] defeated member[{ladder[rank][i_member]}] of the current rank by far, so it goes for the next member if it exists")
  if len(ladder[rank])>i_member+1:
   loop_level="next member"
  else:
   loop_level="next rank"

def start_new_ladder():
 # Start a new ladder
 ladder,unique_id=create_ladder()
 train_ladder(ladder,unique_id)
 return

def ladder_last_id(ladder):
 highest_rank_members=ladder[-1]
 id=random.choice(highest_rank_members)
 print(f"selected member of the highest rank: {id}")
 return int(id)

def continue_existing_ladder():
 # Continue with existing ladder
 ladder=load_ladder()
 unique_id=ladder_last_id(ladder)
 train_ladder(ladder,unique_id)

def test_the_best():
 hi_rank=len(ladder)-1
 random_name=random.choice(ladder[hi_rank])
 hi_model_path=ladder_path+"\\models\\model_"+random_name
 machine=load_model(hi_model_path)
 random_int=random.choice(range(2))
 if random_int==0:
  first="machine"
 else:
  first="human"
 result=play_game(machine,first)
 print(f"machine result= {result}")

# Main function
if __name__ == "__main__":

 # Repeatable random or not
 if b_repeatable_random:
  seed_value=0 #42
  np.random.seed(seed_value)
  random.seed(seed_value)

 verbose
 _int=0
 if verbose=='details':
  verbose_int=1

 ladder=load_ladder()
 if ladder:
  print_ladder(ladder)

 # Display menu
 while True:
  print("Menu:")
  print("1. New train")
  print("2. Continue training")
  print("3. Test the best")
  choice = input("Enter your choice (1/2/3): ")
  if choice == '1':
   print("You selected 'New train'.")
   start_new_ladder()
  elif choice == '2':
   print("You selected 'Continue training'.")
   continue_existing_ladder()
  elif choice == '3':
   print("You selected 'Test the best'.")
   test_the_best()
  else:
   print("Program terminates.")
   break
 
 exit(0)

##########################################################

 # Generate random board states for network input
 """
 batch_size=2 # 9 # number of random board states
 num_simulations=4 # 3  # Number of simulations per board state
 x_train_list=[]
 y_train_list=[]
 for _ in range(batch_size):
  board=random_board()
  print_board(board)
  print("=========")
  i_board=board_to_int(board)
  x_train_list.append(i_board)
  results_for_simulations = [] # List to store results of individual moves
  for _ in range(num_simulations):
   # Run simulations for each possible move for the current board state
   results_for_moves = []  # List to store results for all moves in a simulation
   for row in range(3):
    for col in range(3):
     # Check if the proposed move is valid
     if board[row][col] == '-':
      # The proposed move is valid
      initial_board = copy.deepcopy(board)
      initial_board[row][col] = 'X' # Machine player puts its 'X' piece on [rox,col] position of the board
      end_board=run_simulation(initial_board)
      print(end_board)
      outcome=end_board_value(end_board)
      print(outcome)
      print("---------")
     else:
      outcome=0.5 # Neutral value is used when the move is not legal
     results_for_moves.append(outcome)
   results_for_simulations.append(results_for_moves)
  average_results=[]
  for move_idx in range(9):
   total_result = sum(results_for_simulations[sim_idx][move_idx] for sim_idx in range(num_simulations))
   average_result = total_result / num_simulations
   average_results.append(average_result)
  y_train_list.append(average_results)
 x_train=np.array(x_train_list)
 y_train=np.array(y_train_list)
 """

# Documentation
def documentation():
 # DesireBoard
 #Board game agent
 pass
 
 ## Game
 if game:
  #n x m rectangle\
  #2 players making moves in turns\
  #Machine player's symbol is X\
  #Human player symbol is O\
  #Either of them can be the first player
  pass
 
 ## Neural network
 if neural_network:
  #model=Sequential()\
  #model.add(Dense(n_neurons,activation='relu',input_shape=(1,)))\
  #model.add(Dense(n_rows*n_cols))\
  #The input shape is 1,meaning that every board state will be converted to one integer value.\
  #We take a row-continuous implementation of the board rectangle.\
  #This will be an n x m position integer, and at each position we place the number that identifies the symbol that is occupying that square.\
  #0 - empty square\
  #1 - square occupied by an X\
  #2 - square occupied by an O\
  #So --XO -> 5+1 (to avoid the 0 integer, we add 1)
  pass

 ## Training
 if training:
  #Model fit is run after  each game
  # x_train is the integer representation of the board that the nn sees when it has to chose its next move
  # y_train is the probability distribution for the nn to randomly select the next move from among the possible moves
  # x_train is stored during the game before every nn move
  # y_train is stored during the game after every nn move
  # y_train is updated when the game is ended
  # only that y_train component is updated with the game result, which is the component
  #  of the move that was actually selcted during the game
  # game result is 1.0 if the nn won, 0.0 if the nn lost, 0.5 is the game was a draw
  pass

 return

# Run simulation
"""
def run_simulation(start_board):
 current_board = copy.deepcopy(start_board)
 print_board(current_board)
 current_player = 'O'  # Simulation starts with the opponent player 'O' move
 game_ended = False
 while not game_ended:
  # Create a list of cells that are still empty
  empty_cells = [(i, j) for i in range(3) for j in range(3) if current_board[i][j] == '-']
  # Randomly choose an empty cell for the current player's move
  row, col = random.choice(empty_cells)
  current_board[row][col] = current_player
  print_board(current_board)
  # Check if the game has ended
  game_ended = has_game_ended(current_board, current_player)
  # Switch to the other player for the next move
  current_player = 'X' if current_player == 'O' else 'O'
 return current_board
"""

# test()
"""
def test():
 #chr(183) is middle dot

 board=[
  ['X',' ',' '],
  [' ',' ',' '],
  [' ',' ',' '],]
 print_board(board)
 i_board=board_to_int(board)
 print(i_board)

 board=[
  ['-','-','-'],
  ['-','-','-'],
  ['-','-','O'],]
 print_board(board)
 i_board=board_to_int(board)
 print(i_board)

 board=int_to_board(1)
 print_board(board)
 
 board=int_to_board(12345)
 print_board(board)
"""

# make_human_move
"""
def make_human_move(n, board):
 # Check if n is within the valid range
 if n < '1' or n > '9':
  return "Invalid move. Please choose a number between 1 and 9."

 # Calculate the position for O
 position=(board_dict[n]['index'])%9

 # Check if the calculated position is already occupied
 if board[position] != 'O' and board[position] != 'X' :
  # Place 'O' at the calculated position
  board[position] = 'O'
  return board
 else:
  return "Position {} is already occupied. Please choose a different move.".format(position + 1)
"""

# random_board
"""
def random_board():
 # Initialize an empty board
 board = [['-', '-', '-'],
          ['-', '-', '-'],
          ['-', '-', '-']]
 # Define the maximum number of random moves
 max_moves = 9  # In Tic-Tac-Toe, there are at most 9 moves in a game
 # Start with random player ('X' or 'O')
 current_player = random.choice(['X', 'O'])
 # Set a random number for the number of moves
 #  based on the current player
 if current_player == 'X':
  # If current player is 'X', generate an even number of moves (0 to 8)
  num_moves = random.randrange(0, 9-1+1, 2)
 else:
  # If current player is 'O', generate an odd number of moves (1 to 7)
  num_moves = random.randrange(1, 9-1+1, 2)
 game_ended = False
 for i_move in range(num_moves):
  # Check if the game has ended; if so, break the loop
  if game_ended:
   break
  # Find empty cells where a move can be made
  empty_cells = [(i, j) for i in range(3) for j in range(3) if board[i][j] == '-']
  # Choose a random empty cell
  row, col = random.choice(empty_cells)
  # Make the move for the current player
  board[row][col] = current_player
  # Check if the game has ended after this move
  game_ended = has_game_ended(board, current_player)
  if game_ended:
   # If game ended, then clear the last move
   board[row][col]='-'
  # Toggle the current player for the next move
  current_player = 'X' if current_player == 'O' else 'O'
 # The 'board' now contains a random game state
 return board
"""

# Plot training & validation loss values
"""
def plot_train():
 plt.plot(history.history['loss'])
 plt.title('Model Loss')
 plt.xlabel('Epoch')
 plt.ylabel('Loss')
 plt.legend(['Train'], loc='upper left')
 plt.show()
"""
