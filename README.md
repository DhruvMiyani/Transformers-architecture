<a name="br1"></a> 

PA 2 Report

Dhruv Miyani

[My](https://colab.research.google.com/drive/12FbZJOLqJY18o9J1JWZVX5UJUKMVZbUd?usp=sharing)[ ](https://colab.research.google.com/drive/12FbZJOLqJY18o9J1JWZVX5UJUKMVZbUd?usp=sharing)[python](https://colab.research.google.com/drive/12FbZJOLqJY18o9J1JWZVX5UJUKMVZbUd?usp=sharing)[ ](https://colab.research.google.com/drive/12FbZJOLqJY18o9J1JWZVX5UJUKMVZbUd?usp=sharing)[notebook](https://colab.research.google.com/drive/12FbZJOLqJY18o9J1JWZVX5UJUKMVZbUd?usp=sharing)

Why Transformer Architecture Cons RNN in NLP

Cons of RNN in NLP:

• Slow for long sequences

• Losing Gradients

To solve this, the Transformer architecture has an encoder-decoder structure but does not rely on RNN in

order to generate an output.

1



<a name="br2"></a> 

1 Multi - Head Attention Cell 6 in notebook

We have QKV:

(a) Query

(b) Key

(c) Value

These are input matrices.

Input Linear Projections (W , W , W ) : In the diagram, the inputs pass through linear layers to create Q,

q

k

v

K, and V (queries, keys, and values). The code deﬁnes three linear transformations self.W q, self.W k, and

self.W v for this purpose.

self attention

The self-attention mechanism in a Transformer model is calculated as follows:

ꢀ

ꢁ

QK<sup>T</sup>

Attention(Q, K, V ) = softmax

√

V

d<sub>k</sub>

where:

• Q is the query matrix with dimensions (n, d<sub>k</sub>),

• K is the key matrix with dimensions (n, d<sub>k</sub>),

• V is the value matrix with dimensions (n, d<sub>v</sub>),

• d<sub>k</sub> is the dimensionality of the keys and queries, typically the same as the dimensionality of the model.

Splitting Heads (split heads): The linearly transformed Q, K, and V are then split into multiple heads

(head1, head2, ..., head h in the diagram). The code’s split heads function reshapes the input tensor to ac-

commodate the multi-head structure.

2



<a name="br3"></a> 

Scaled Dot-Product Attention (scaled dot product attention): Each head computes attention using the

formula shown in the diagram:

ꢀ

ꢁ

QK<sup>T</sup>

Attention(Q, K, V ) = softmax

√

V.

d<sub>k</sub>

In the code, this formula is implemented in the scaled dot product attention function, where d<sub>k</sub> is the

dimension of K (or Q since they are the same).

Concatenating Heads (combine heads): After computing attention for each head, the results are concate-

nated back to form a single matrix H in the diagram. In the code, the combine heads function is the inverse of

split heads, reverting the multi-head structure back to the original embedding dimension.

Output Linear Projection (W<sub>o</sub>): Finally, the concatenated heads pass through another linear layer to pro-

duce the ﬁnal output MH −A in the diagram. The code has a linear transformation self.W o that corresponds

to this ﬁnal step.

Forward Pass (forward): The forward method in the code orchestrates the entire process: linear projections,

splitting into heads, applying scaled dot-product attention for each head, concatenating the heads’ outputs,

and applying the ﬁnal linear projection.

2 positional encoding cell 8

Positional Encoding is added to give the model information about the position of words in a sequence.

The PositionalEncoding class in the code snippet implements this with a matrix pe of dimensions [max seq length,

d model], using sine and cosine functions with different frequencies:

ꢂ

ꢃ

pos

PE(pos, 2i) = sin

10000<sup>2i/d</sup>model

pos

ꢂ

ꢃ

PE(pos, 2i + 1) = cos

10000<sup>2i/d</sup>model

The position tensor is created with values ranging from 0 to max seq length - 1, reshaped to [max seq length,

1].

The div term is calculated based on the model dimension and position, creating a unique encoding for each

position.

The pe matrix is registered as a buffer to be persistent and not updated during training.

During the forward pass, the positional encodings are added to the input embeddings, leveraging broad-

casting to apply across the batch.

3



<a name="br4"></a> 

3 Training

During the training of a Transformer model, the following steps are executed, corresponding to the code snip-

pet and the images provided:

1\. model.train() sets the model to training mode, enabling speciﬁc layers like dropout layers to function

appropriately during training.

2\. output = model(src\_data, tgt\_data[:, :-1]) executes the forward pass, processing the input through

the encoder and then the decoder.

3\. The model output is projected to the vocabulary size by an internal linear layer and transformed into

probabilities using a softmax layer.

4\. loss = criterion(output, tgt) calculates the cross-entropy loss between the predicted probabilities

and the actual targets.

5\. loss.backward() computes the gradients of the loss with respect to the model parameters.

6\. nn.utils.clip\_grad\_norm\_(model.parameters(), grad\_clip) performs gradient clipping to mitigate

exploding gradients.

7\. optimizer.step() updates the model parameters based on the gradients.

8\. Hyperparameters like batch\_size, grad\_clip, and the optimizer settings are deﬁned to control the train-

ing loop’s behavior.

Each epoch represents a full iteration over the training dataset, with the model learning to map the input

sequence to the target sequence.

4 Implementation details

I faced challenges while running the modal for 20 epochs , then i switched to GPU, then it ran faster

I was stuck in between for 2 days at vocab building , ﬁnally i got help from piazza post.

in this assignment i learned multi head attention , it was not clear earlier ,now i know practically.

I tried to relate code with diagram mentioned above, this process help me understand code better.

5 Output

”Ein kleiner Junge spielt draußen mit einem Ball.” German for ”A little boy playing outside with a ball.”

Translated sentence: ” A young boy is playing with a toy ”

for hyperperameter table 3 result : ”A group of people are playing”

4



<a name="br5"></a> 

Table 1: Hyperparameters and Results for Transformer Model (Type 2)

Epoch Hyperparameter Train Loss Val Loss

1

Source Vocabulary Size (src vocab size)

Target Vocabulary Size (tgt vocab size)

Embedding Dimension (d model)

Number of Encoder and Decoder Layers (N)

Number of Attention Heads (num heads)

Dimension of Feed Forward Networks (d ff)

5000

5000

512

6

5000

5000

512

6

8

8

2048

2048

100

Maximum Sequence Length (max seq length) 100

Dropout Rate (dropout)

Padding Token Index (pad idx)

Train Loss

Val Loss

Train Loss

Val Loss

Train Loss

Val Loss

Train Loss

Val Loss

Train Loss

Val Loss

Train Loss

Val Loss

Train Loss

Val Loss

Train Loss

Val Loss

0\.1

0

0\.1

0

2

5\.689

4\.876

4\.694

4\.426

4\.106

3\.894

3\.740

3\.615

3\.521

3\.442

3\.376

3\.313

3\.243

3\.181

3\.121

3\.066

3\.010

2\.956

2\.909

2\.864

5\.015

4\.796

4\.610

4\.252

4\.016

3\.874

3\.733

3\.666

3\.596

3\.555

3\.495

3\.478

3\.396

3\.361

3\.314

3\.266

3\.238

3\.218

3\.176

3\.152

3

4

5

6

7

8

9

10

11

Train Loss

Val Loss

Train Loss

Val Loss

Table 2: Hyperparameters for Transformer Model (Type 3)

Hyperparameter

Value

5000

5000

128

4

Source Vocabulary Size (src vocab size)

Target Vocabulary Size (tgt vocab size)

Embedding Dimension (d model)

Number of Encoder and Decoder Layers (N)

Number of Attention Heads (num heads)

Dimension of Feed Forward Networks (d ff)

4

512

Maximum Sequence Length (max seq length) 100

Dropout Rate (dropout)

Padding Token Index (pad idx)

0\.1

0\.1

Table 3: Hyperparameters for Transformer Model (Type 2)

Hyperparameter

Value

5000

5000

512

6

Source Vocabulary Size (src vocab size)

Target Vocabulary Size (tgt vocab size)

Embedding Dimension (d model)

Number of Encoder and Decoder Layers (N)

Number of Attention Heads (num heads)

Dimension of Feed Forward Networks (d ff)

8

2048

Maximum Sequence Length (max seq length) 100

Dropout Rate (dropout)

Padding Token Index (pad idx)

0\.1

0

5



<a name="br6"></a> 

Table 4: Hyperparameters for Transformer Model (Type 3)

Hyperparameter

Value

5000

5000

128

4

Source Vocabulary Size (src vocab size)

Target Vocabulary Size (tgt vocab size)

Embedding Dimension (d model)

Number of Encoder and Decoder Layers (N)

Number of Attention Heads (num heads)

Dimension of Feed Forward Networks (d ff)

4

512

Maximum Sequence Length (max seq length) 100

Dropout Rate (dropout)

Padding Token Index (pad idx)

0\.1

0\.1

Table 5: Results for Transformer Model (Type 3)

Epoch Train Loss Val Loss

1

2

3

4

5

6

7

8

5\.691

4\.893

4\.701

4\.442

4\.105

3\.892

3\.749

3\.632

3\.531

3\.453

3\.386

3\.322

3\.262

3\.207

3\.150

3\.095

3\.038

2\.991

2\.939

2\.890

4\.995

4\.792

4\.616

4\.258

4\.032

3\.861

3\.770

3\.700

3\.605

3\.563

3\.516

3\.469

3\.440

3\.392

3\.350

3\.298

3\.269

3\.246

3\.208

3\.181

9

10

11

12

13

14

15

16

17

18

19

20

6

[DS_5983_LLM (5) (1).md](https://github.com/DhruvMiyani/Transformers-architecture/files/14875810/DS_5983_LLM.5.1.md)

