??)
?:?9
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	??
?
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
?
AsString

input"T

output"
Ttype:
2		
"
	precisionint?????????"

scientificbool( "
shortestbool( "
widthint?????????"
fillstring 
x
Assign
ref"T?

value"T

output_ref"T?"	
Ttype"
validate_shapebool("
use_lockingbool(?
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
S
	Bucketize

input"T

output"
Ttype:
2	"

boundarieslist(float)
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
B
Equal
x"T
y"T
z
"
Ttype:
2	
?
W

ExpandDims

input"T
dim"Tdim
output"T"	
Ttype"
Tdimtype0:
2	
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
p
GatherNd
params"Tparams
indices"Tindices
output"Tparams"
Tparamstype"
Tindicestype:
2	
?
GatherV2
params"Tparams
indices"Tindices
axis"Taxis
output"Tparams"

batch_dimsint "
Tparamstype"
Tindicestype:
2	"
Taxistype:
2	
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
?
HashTableV2
table_handle"
	containerstring "
shared_namestring "!
use_node_name_sharingbool( "
	key_dtypetype"
value_dtypetype?
V
HistogramSummary
tag
values"T
summary"
Ttype0:
2	
.
Identity

input"T
output"T"	
Ttype
?
	LessEqual
x"T
y"T
z
"
Ttype:
2	
w
LookupTableFindV2
table_handle
keys"Tin
default_value"Tout
values"Tout"
Tintype"
Touttype?
b
LookupTableImportV2
table_handle
keys"Tin
values"Tout"
Tintype"
Touttype?
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
N
Merge
inputs"T*N
output"T
value_index"	
Ttype"
Nint(0
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?

NoOp
E
NotEqual
x"T
y"T
z
"
Ttype:
2	
?
?
OneHot
indices"TI	
depth
on_value"T
	off_value"T
output"T"
axisint?????????"	
Ttype"
TItype0	:
2	
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
?
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
@
ReadVariableOp
resource
value"dtype"
dtypetype?
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
?
ResourceGather
resource
indices"Tindices
output"dtype"

batch_dimsint "
validate_indicesbool("
dtypetype"
Tindicestype:
2	?
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
P
ScalarSummary
tags
values"T
summary"
Ttype:
2	
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
0
Sigmoid
x"T
y"T"
Ttype:

2
O
Size

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
a
Slice

input"T
begin"Index
size"Index
output"T"	
Ttype"
Indextype:
2	
9
Softmax
logits"T
softmax"T"
Ttype:
2
?
SparseCross
indices	*N
values2sparse_types
shapes	*N
dense_inputs2dense_types
output_indices	
output_values"out_type
output_shape	"

Nint("
hashed_outputbool"
num_bucketsint("
hash_keyint"$
sparse_types
list(type)(:
2	"#
dense_types
list(type)(:
2	"
out_typetype:
2	"
internal_typetype:
2	
?
SparseFillEmptyRows
indices	
values"T
dense_shape	
default_value"T
output_indices	
output_values"T
empty_row_indicator

reverse_index_map	"	
Ttype
h
SparseReshape
input_indices	
input_shape	
	new_shape	
output_indices	
output_shape	
z
SparseSegmentMean	
data"T
indices"Tidx
segment_ids
output"T"
Ttype:
2"
Tidxtype0:
2	
?
SparseSegmentSum	
data"T
indices"Tidx
segment_ids
output"T"
Ttype:
2	"
Tidxtype0:
2	
?
SparseToDense
sparse_indices"Tindices
output_shape"Tindices
sparse_values"T
default_value"T

dense"T"
validate_indicesbool("	
Ttype"
Tindicestype:
2	
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
G
StringToHashBucketFast	
input

output	"
num_bucketsint(0
:
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
M
Switch	
data"T
pred

output_false"T
output_true"T"	
Ttype
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
?
TruncatedNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
P
Unique
x"T
y"T
idx"out_idx"	
Ttype"
out_idxtype0:
2	
q
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape?
9
VarIsInitializedOp
resource
is_initialized
?
s

VariableV2
ref"dtype?"
shapeshape"
dtypetype"
	containerstring "
shared_namestring ?
E
Where

input"T	
index	"%
Ttype0
:
2	

&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.14.02v1.14.0-rc1-22-gaf24dc91b58??#

global_step/Initializer/zerosConst*
value	B	 R *
_class
loc:@global_step*
dtype0	*
_output_shapes
: 
k
global_step
VariableV2*
dtype0	*
_output_shapes
: *
shape: *
_class
loc:@global_step
?
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
_output_shapes
: *
T0	*
_class
loc:@global_step
j
global_step/readIdentityglobal_step*
T0	*
_class
loc:@global_step*
_output_shapes
: 
f
PlaceholderPlaceholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
h
Placeholder_1Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
h
Placeholder_2Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
h
Placeholder_4Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
h
Placeholder_5Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
h
Placeholder_6Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
h
Placeholder_7Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
h
Placeholder_8Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
h
Placeholder_9Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
i
Placeholder_10Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
i
Placeholder_11Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
i
Placeholder_12Placeholder*
dtype0*#
_output_shapes
:?????????*
shape:?????????
?
dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"d      *o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0
?
~dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0
?
?dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
valueB
 *??>*o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0*
dtype0*
_output_shapes
: 
?
?dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormaldnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
dtype0*
_output_shapes

:d*
T0*o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0
?
}dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMul?dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal?dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0*
_output_shapes

:d
?
ydnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Initializer/truncated_normalAdd}dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Initializer/truncated_normal/mul~dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
_output_shapes

:d*
T0*o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0
?
\dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape
:d*m
shared_name^\dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0*o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0
?
}dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp\dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0*
_output_shapes
: 
?
cdnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/AssignAssignVariableOp\dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0ydnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Initializer/truncated_normal*o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0*
dtype0
?
pdnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Read/ReadVariableOpReadVariableOp\dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0*o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0*
dtype0*
_output_shapes

:d
?
{dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/shapeConst*
dtype0*
_output_shapes
:*
valueB"d      *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0
?
zdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/meanConst*
dtype0*
_output_shapes
: *
valueB
 *    *k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0
?
|dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddevConst*
dtype0*
_output_shapes
: *
valueB
 *??>*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormalTruncatedNormal{dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/shape*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
dtype0*
_output_shapes

:d
?
ydnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulMul?dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/TruncatedNormal|dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/stddev*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes

:d
?
udnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normalAddydnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/mulzdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal/mean*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes

:d
?
Xdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape
:d*i
shared_nameZXdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0
?
ydnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpXdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
_output_shapes
: 
?
_dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/AssignAssignVariableOpXdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0udnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
dtype0
?
ldnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Read/ReadVariableOpReadVariableOpXdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
dtype0*
_output_shapes

:d
?
=dnn/input_from_feature_columns/input_layer/age/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
?????????
?
9dnn/input_from_feature_columns/input_layer/age/ExpandDims
ExpandDimsPlaceholder_8=dnn/input_from_feature_columns/input_layer/age/ExpandDims/dim*'
_output_shapes
:?????????*
T0
?
4dnn/input_from_feature_columns/input_layer/age/ShapeShape9dnn/input_from_feature_columns/input_layer/age/ExpandDims*
T0*
_output_shapes
:
?
Bdnn/input_from_feature_columns/input_layer/age/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Ddnn/input_from_feature_columns/input_layer/age/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Ddnn/input_from_feature_columns/input_layer/age/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
<dnn/input_from_feature_columns/input_layer/age/strided_sliceStridedSlice4dnn/input_from_feature_columns/input_layer/age/ShapeBdnn/input_from_feature_columns/input_layer/age/strided_slice/stackDdnn/input_from_feature_columns/input_layer/age/strided_slice/stack_1Ddnn/input_from_feature_columns/input_layer/age/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
?
>dnn/input_from_feature_columns/input_layer/age/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
<dnn/input_from_feature_columns/input_layer/age/Reshape/shapePack<dnn/input_from_feature_columns/input_layer/age/strided_slice>dnn/input_from_feature_columns/input_layer/age/Reshape/shape/1*
N*
_output_shapes
:*
T0
?
6dnn/input_from_feature_columns/input_layer/age/ReshapeReshape9dnn/input_from_feature_columns/input_layer/age/ExpandDims<dnn/input_from_feature_columns/input_layer/age/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Fdnn/input_from_feature_columns/input_layer/capital_gain/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Bdnn/input_from_feature_columns/input_layer/capital_gain/ExpandDims
ExpandDimsPlaceholder_10Fdnn/input_from_feature_columns/input_layer/capital_gain/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
=dnn/input_from_feature_columns/input_layer/capital_gain/ShapeShapeBdnn/input_from_feature_columns/input_layer/capital_gain/ExpandDims*
T0*
_output_shapes
:
?
Kdnn/input_from_feature_columns/input_layer/capital_gain/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
?
Mdnn/input_from_feature_columns/input_layer/capital_gain/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
?
Mdnn/input_from_feature_columns/input_layer/capital_gain/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Ednn/input_from_feature_columns/input_layer/capital_gain/strided_sliceStridedSlice=dnn/input_from_feature_columns/input_layer/capital_gain/ShapeKdnn/input_from_feature_columns/input_layer/capital_gain/strided_slice/stackMdnn/input_from_feature_columns/input_layer/capital_gain/strided_slice/stack_1Mdnn/input_from_feature_columns/input_layer/capital_gain/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
?
Gdnn/input_from_feature_columns/input_layer/capital_gain/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Ednn/input_from_feature_columns/input_layer/capital_gain/Reshape/shapePackEdnn/input_from_feature_columns/input_layer/capital_gain/strided_sliceGdnn/input_from_feature_columns/input_layer/capital_gain/Reshape/shape/1*
N*
_output_shapes
:*
T0
?
?dnn/input_from_feature_columns/input_layer/capital_gain/ReshapeReshapeBdnn/input_from_feature_columns/input_layer/capital_gain/ExpandDimsEdnn/input_from_feature_columns/input_layer/capital_gain/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Fdnn/input_from_feature_columns/input_layer/capital_loss/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Bdnn/input_from_feature_columns/input_layer/capital_loss/ExpandDims
ExpandDimsPlaceholder_11Fdnn/input_from_feature_columns/input_layer/capital_loss/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
=dnn/input_from_feature_columns/input_layer/capital_loss/ShapeShapeBdnn/input_from_feature_columns/input_layer/capital_loss/ExpandDims*
T0*
_output_shapes
:
?
Kdnn/input_from_feature_columns/input_layer/capital_loss/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
?
Mdnn/input_from_feature_columns/input_layer/capital_loss/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Mdnn/input_from_feature_columns/input_layer/capital_loss/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Ednn/input_from_feature_columns/input_layer/capital_loss/strided_sliceStridedSlice=dnn/input_from_feature_columns/input_layer/capital_loss/ShapeKdnn/input_from_feature_columns/input_layer/capital_loss/strided_slice/stackMdnn/input_from_feature_columns/input_layer/capital_loss/strided_slice/stack_1Mdnn/input_from_feature_columns/input_layer/capital_loss/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
?
Gdnn/input_from_feature_columns/input_layer/capital_loss/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Ednn/input_from_feature_columns/input_layer/capital_loss/Reshape/shapePackEdnn/input_from_feature_columns/input_layer/capital_loss/strided_sliceGdnn/input_from_feature_columns/input_layer/capital_loss/Reshape/shape/1*
N*
_output_shapes
:*
T0
?
?dnn/input_from_feature_columns/input_layer/capital_loss/ReshapeReshapeBdnn/input_from_feature_columns/input_layer/capital_loss/ExpandDimsEdnn/input_from_feature_columns/input_layer/capital_loss/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Mdnn/input_from_feature_columns/input_layer/education_indicator/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
?????????
?
Idnn/input_from_feature_columns/input_layer/education_indicator/ExpandDims
ExpandDimsPlaceholder_2Mdnn/input_from_feature_columns/input_layer/education_indicator/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
]dnn/input_from_feature_columns/input_layer/education_indicator/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
Wdnn/input_from_feature_columns/input_layer/education_indicator/to_sparse_input/NotEqualNotEqualIdnn/input_from_feature_columns/input_layer/education_indicator/ExpandDims]dnn/input_from_feature_columns/input_layer/education_indicator/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:?????????
?
Vdnn/input_from_feature_columns/input_layer/education_indicator/to_sparse_input/indicesWhereWdnn/input_from_feature_columns/input_layer/education_indicator/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Udnn/input_from_feature_columns/input_layer/education_indicator/to_sparse_input/valuesGatherNdIdnn/input_from_feature_columns/input_layer/education_indicator/ExpandDimsVdnn/input_from_feature_columns/input_layer/education_indicator/to_sparse_input/indices*#
_output_shapes
:?????????*
Tindices0	*
Tparams0
?
Zdnn/input_from_feature_columns/input_layer/education_indicator/to_sparse_input/dense_shapeShapeIdnn/input_from_feature_columns/input_layer/education_indicator/ExpandDims*
T0*
out_type0	*
_output_shapes
:
?
Udnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/ConstConst*
dtype0*
_output_shapes
:*?
value?B?B
 BachelorsB HS-gradB 11thB MastersB 9thB Some-collegeB Assoc-acdmB
 Assoc-vocB 7th-8thB
 DoctorateB Prof-schoolB 5th-6thB 10thB 1st-4thB
 PreschoolB 12th
?
Tdnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
?
[dnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
?
[dnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
Udnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/rangeRange[dnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/range/startTdnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/Size[dnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/range/delta*
_output_shapes
:
?
Tdnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/CastCastUdnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/range*

SrcT0*
_output_shapes
:*

DstT0	
?
`dnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/ConstConst*
dtype0	*
_output_shapes
: *
valueB	 R
?????????
?
ednn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/hash_tableHashTableV2*
value_dtype0	*
	key_dtype0*
_output_shapes
: 
?
ydnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2ednn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/hash_tableUdnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/ConstTdnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/Cast*	
Tin0*

Tout0	
?
bdnn/input_from_feature_columns/input_layer/education_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2ednn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/hash_tableUdnn/input_from_feature_columns/input_layer/education_indicator/to_sparse_input/values`dnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/Const*#
_output_shapes
:?????????*	
Tin0*

Tout0	
?
Zdnn/input_from_feature_columns/input_layer/education_indicator/SparseToDense/default_valueConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
Ldnn/input_from_feature_columns/input_layer/education_indicator/SparseToDenseSparseToDenseVdnn/input_from_feature_columns/input_layer/education_indicator/to_sparse_input/indicesZdnn/input_from_feature_columns/input_layer/education_indicator/to_sparse_input/dense_shapebdnn/input_from_feature_columns/input_layer/education_indicator/hash_table_Lookup/LookupTableFindV2Zdnn/input_from_feature_columns/input_layer/education_indicator/SparseToDense/default_value*'
_output_shapes
:?????????*
Tindices0	*
T0	
?
Ldnn/input_from_feature_columns/input_layer/education_indicator/one_hot/ConstConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Ndnn/input_from_feature_columns/input_layer/education_indicator/one_hot/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ldnn/input_from_feature_columns/input_layer/education_indicator/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
?
Odnn/input_from_feature_columns/input_layer/education_indicator/one_hot/on_valueConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Pdnn/input_from_feature_columns/input_layer/education_indicator/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Fdnn/input_from_feature_columns/input_layer/education_indicator/one_hotOneHotLdnn/input_from_feature_columns/input_layer/education_indicator/SparseToDenseLdnn/input_from_feature_columns/input_layer/education_indicator/one_hot/depthOdnn/input_from_feature_columns/input_layer/education_indicator/one_hot/on_valuePdnn/input_from_feature_columns/input_layer/education_indicator/one_hot/off_value*+
_output_shapes
:?????????*
T0
?
Tdnn/input_from_feature_columns/input_layer/education_indicator/Sum/reduction_indicesConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Bdnn/input_from_feature_columns/input_layer/education_indicator/SumSumFdnn/input_from_feature_columns/input_layer/education_indicator/one_hotTdnn/input_from_feature_columns/input_layer/education_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:?????????
?
Ddnn/input_from_feature_columns/input_layer/education_indicator/ShapeShapeBdnn/input_from_feature_columns/input_layer/education_indicator/Sum*
_output_shapes
:*
T0
?
Rdnn/input_from_feature_columns/input_layer/education_indicator/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Tdnn/input_from_feature_columns/input_layer/education_indicator/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
?
Tdnn/input_from_feature_columns/input_layer/education_indicator/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Ldnn/input_from_feature_columns/input_layer/education_indicator/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/education_indicator/ShapeRdnn/input_from_feature_columns/input_layer/education_indicator/strided_slice/stackTdnn/input_from_feature_columns/input_layer/education_indicator/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/education_indicator/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
?
Ndnn/input_from_feature_columns/input_layer/education_indicator/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Ldnn/input_from_feature_columns/input_layer/education_indicator/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/education_indicator/strided_sliceNdnn/input_from_feature_columns/input_layer/education_indicator/Reshape/shape/1*
N*
_output_shapes
:*
T0
?
Fdnn/input_from_feature_columns/input_layer/education_indicator/ReshapeReshapeBdnn/input_from_feature_columns/input_layer/education_indicator/SumLdnn/input_from_feature_columns/input_layer/education_indicator/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Gdnn/input_from_feature_columns/input_layer/education_num/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Cdnn/input_from_feature_columns/input_layer/education_num/ExpandDims
ExpandDimsPlaceholder_9Gdnn/input_from_feature_columns/input_layer/education_num/ExpandDims/dim*'
_output_shapes
:?????????*
T0
?
>dnn/input_from_feature_columns/input_layer/education_num/ShapeShapeCdnn/input_from_feature_columns/input_layer/education_num/ExpandDims*
T0*
_output_shapes
:
?
Ldnn/input_from_feature_columns/input_layer/education_num/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Ndnn/input_from_feature_columns/input_layer/education_num/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Ndnn/input_from_feature_columns/input_layer/education_num/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Fdnn/input_from_feature_columns/input_layer/education_num/strided_sliceStridedSlice>dnn/input_from_feature_columns/input_layer/education_num/ShapeLdnn/input_from_feature_columns/input_layer/education_num/strided_slice/stackNdnn/input_from_feature_columns/input_layer/education_num/strided_slice/stack_1Ndnn/input_from_feature_columns/input_layer/education_num/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Hdnn/input_from_feature_columns/input_layer/education_num/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Fdnn/input_from_feature_columns/input_layer/education_num/Reshape/shapePackFdnn/input_from_feature_columns/input_layer/education_num/strided_sliceHdnn/input_from_feature_columns/input_layer/education_num/Reshape/shape/1*
T0*
N*
_output_shapes
:
?
@dnn/input_from_feature_columns/input_layer/education_num/ReshapeReshapeCdnn/input_from_feature_columns/input_layer/education_num/ExpandDimsFdnn/input_from_feature_columns/input_layer/education_num/Reshape/shape*'
_output_shapes
:?????????*
T0
?
Jdnn/input_from_feature_columns/input_layer/gender_indicator/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Fdnn/input_from_feature_columns/input_layer/gender_indicator/ExpandDims
ExpandDimsPlaceholderJdnn/input_from_feature_columns/input_layer/gender_indicator/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
Zdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
Tdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/NotEqualNotEqualFdnn/input_from_feature_columns/input_layer/gender_indicator/ExpandDimsZdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:?????????
?
Sdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/indicesWhereTdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Rdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/valuesGatherNdFdnn/input_from_feature_columns/input_layer/gender_indicator/ExpandDimsSdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/indices*#
_output_shapes
:?????????*
Tindices0	*
Tparams0
?
Wdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/dense_shapeShapeFdnn/input_from_feature_columns/input_layer/gender_indicator/ExpandDims*
T0*
out_type0	*
_output_shapes
:
?
Odnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/ConstConst*
dtype0*
_output_shapes
:*#
valueBB FemaleB Male
?
Ndnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
?
Udnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
?
Udnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
Odnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/rangeRangeUdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/range/startNdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/SizeUdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/range/delta*
_output_shapes
:
?
Ndnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/CastCastOdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/range*

SrcT0*
_output_shapes
:*

DstT0	
?
Zdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_table/ConstConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
_dnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_table/hash_tableHashTableV2*
value_dtype0	*
	key_dtype0*
_output_shapes
: 
?
sdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2_dnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_table/hash_tableOdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/ConstNdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/Cast*	
Tin0*

Tout0	
?
_dnn/input_from_feature_columns/input_layer/gender_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2_dnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_table/hash_tableRdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/valuesZdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_table/Const*#
_output_shapes
:?????????*	
Tin0*

Tout0	
?
Wdnn/input_from_feature_columns/input_layer/gender_indicator/SparseToDense/default_valueConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
Idnn/input_from_feature_columns/input_layer/gender_indicator/SparseToDenseSparseToDenseSdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/indicesWdnn/input_from_feature_columns/input_layer/gender_indicator/to_sparse_input/dense_shape_dnn/input_from_feature_columns/input_layer/gender_indicator/hash_table_Lookup/LookupTableFindV2Wdnn/input_from_feature_columns/input_layer/gender_indicator/SparseToDense/default_value*
Tindices0	*
T0	*'
_output_shapes
:?????????
?
Idnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
Kdnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/Const_1Const*
dtype0*
_output_shapes
: *
valueB
 *    
?
Idnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
?
Ldnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/on_valueConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Mdnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/off_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
?
Cdnn/input_from_feature_columns/input_layer/gender_indicator/one_hotOneHotIdnn/input_from_feature_columns/input_layer/gender_indicator/SparseToDenseIdnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/depthLdnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/on_valueMdnn/input_from_feature_columns/input_layer/gender_indicator/one_hot/off_value*
T0*+
_output_shapes
:?????????
?
Qdnn/input_from_feature_columns/input_layer/gender_indicator/Sum/reduction_indicesConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?
?dnn/input_from_feature_columns/input_layer/gender_indicator/SumSumCdnn/input_from_feature_columns/input_layer/gender_indicator/one_hotQdnn/input_from_feature_columns/input_layer/gender_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:?????????
?
Adnn/input_from_feature_columns/input_layer/gender_indicator/ShapeShape?dnn/input_from_feature_columns/input_layer/gender_indicator/Sum*
T0*
_output_shapes
:
?
Odnn/input_from_feature_columns/input_layer/gender_indicator/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Qdnn/input_from_feature_columns/input_layer/gender_indicator/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Qdnn/input_from_feature_columns/input_layer/gender_indicator/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Idnn/input_from_feature_columns/input_layer/gender_indicator/strided_sliceStridedSliceAdnn/input_from_feature_columns/input_layer/gender_indicator/ShapeOdnn/input_from_feature_columns/input_layer/gender_indicator/strided_slice/stackQdnn/input_from_feature_columns/input_layer/gender_indicator/strided_slice/stack_1Qdnn/input_from_feature_columns/input_layer/gender_indicator/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
?
Kdnn/input_from_feature_columns/input_layer/gender_indicator/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :
?
Idnn/input_from_feature_columns/input_layer/gender_indicator/Reshape/shapePackIdnn/input_from_feature_columns/input_layer/gender_indicator/strided_sliceKdnn/input_from_feature_columns/input_layer/gender_indicator/Reshape/shape/1*
N*
_output_shapes
:*
T0
?
Cdnn/input_from_feature_columns/input_layer/gender_indicator/ReshapeReshape?dnn/input_from_feature_columns/input_layer/gender_indicator/SumIdnn/input_from_feature_columns/input_layer/gender_indicator/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Hdnn/input_from_feature_columns/input_layer/hours_per_week/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Ddnn/input_from_feature_columns/input_layer/hours_per_week/ExpandDims
ExpandDimsPlaceholder_12Hdnn/input_from_feature_columns/input_layer/hours_per_week/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/hours_per_week/ShapeShapeDdnn/input_from_feature_columns/input_layer/hours_per_week/ExpandDims*
T0*
_output_shapes
:
?
Mdnn/input_from_feature_columns/input_layer/hours_per_week/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
?
Odnn/input_from_feature_columns/input_layer/hours_per_week/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Odnn/input_from_feature_columns/input_layer/hours_per_week/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Gdnn/input_from_feature_columns/input_layer/hours_per_week/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/hours_per_week/ShapeMdnn/input_from_feature_columns/input_layer/hours_per_week/strided_slice/stackOdnn/input_from_feature_columns/input_layer/hours_per_week/strided_slice/stack_1Odnn/input_from_feature_columns/input_layer/hours_per_week/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
?
Idnn/input_from_feature_columns/input_layer/hours_per_week/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Gdnn/input_from_feature_columns/input_layer/hours_per_week/Reshape/shapePackGdnn/input_from_feature_columns/input_layer/hours_per_week/strided_sliceIdnn/input_from_feature_columns/input_layer/hours_per_week/Reshape/shape/1*
T0*
N*
_output_shapes
:
?
Adnn/input_from_feature_columns/input_layer/hours_per_week/ReshapeReshapeDdnn/input_from_feature_columns/input_layer/hours_per_week/ExpandDimsGdnn/input_from_feature_columns/input_layer/hours_per_week/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Rdnn/input_from_feature_columns/input_layer/marital_status_indicator/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
?????????
?
Ndnn/input_from_feature_columns/input_layer/marital_status_indicator/ExpandDims
ExpandDimsPlaceholder_3Rdnn/input_from_feature_columns/input_layer/marital_status_indicator/ExpandDims/dim*'
_output_shapes
:?????????*
T0
?
bdnn/input_from_feature_columns/input_layer/marital_status_indicator/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
\dnn/input_from_feature_columns/input_layer/marital_status_indicator/to_sparse_input/NotEqualNotEqualNdnn/input_from_feature_columns/input_layer/marital_status_indicator/ExpandDimsbdnn/input_from_feature_columns/input_layer/marital_status_indicator/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:?????????
?
[dnn/input_from_feature_columns/input_layer/marital_status_indicator/to_sparse_input/indicesWhere\dnn/input_from_feature_columns/input_layer/marital_status_indicator/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Zdnn/input_from_feature_columns/input_layer/marital_status_indicator/to_sparse_input/valuesGatherNdNdnn/input_from_feature_columns/input_layer/marital_status_indicator/ExpandDims[dnn/input_from_feature_columns/input_layer/marital_status_indicator/to_sparse_input/indices*#
_output_shapes
:?????????*
Tindices0	*
Tparams0
?
_dnn/input_from_feature_columns/input_layer/marital_status_indicator/to_sparse_input/dense_shapeShapeNdnn/input_from_feature_columns/input_layer/marital_status_indicator/ExpandDims*
T0*
out_type0	*
_output_shapes
:
?
_dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/ConstConst*?
value|BzB Married-civ-spouseB	 DivorcedB Married-spouse-absentB Never-marriedB
 SeparatedB Married-AF-spouseB Widowed*
dtype0*
_output_shapes
:
?
^dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
?
ednn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
?
ednn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
_dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/rangeRangeednn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/range/start^dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/Sizeednn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/range/delta*
_output_shapes
:
?
^dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/CastCast_dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/range*

SrcT0*
_output_shapes
:*

DstT0	
?
jdnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/ConstConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
odnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/hash_tableHashTableV2*
value_dtype0	*
	key_dtype0*
_output_shapes
: 
?
?dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2odnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/hash_table_dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/Const^dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/Cast*	
Tin0*

Tout0	
?
gdnn/input_from_feature_columns/input_layer/marital_status_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2odnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/hash_tableZdnn/input_from_feature_columns/input_layer/marital_status_indicator/to_sparse_input/valuesjdnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/Const*#
_output_shapes
:?????????*	
Tin0*

Tout0	
?
_dnn/input_from_feature_columns/input_layer/marital_status_indicator/SparseToDense/default_valueConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
Qdnn/input_from_feature_columns/input_layer/marital_status_indicator/SparseToDenseSparseToDense[dnn/input_from_feature_columns/input_layer/marital_status_indicator/to_sparse_input/indices_dnn/input_from_feature_columns/input_layer/marital_status_indicator/to_sparse_input/dense_shapegdnn/input_from_feature_columns/input_layer/marital_status_indicator/hash_table_Lookup/LookupTableFindV2_dnn/input_from_feature_columns/input_layer/marital_status_indicator/SparseToDense/default_value*
T0	*'
_output_shapes
:?????????*
Tindices0	
?
Qdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hot/ConstConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Sdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hot/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Qdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
?
Tdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hot/on_valueConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Udnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Kdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hotOneHotQdnn/input_from_feature_columns/input_layer/marital_status_indicator/SparseToDenseQdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hot/depthTdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hot/on_valueUdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hot/off_value*
T0*+
_output_shapes
:?????????
?
Ydnn/input_from_feature_columns/input_layer/marital_status_indicator/Sum/reduction_indicesConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Gdnn/input_from_feature_columns/input_layer/marital_status_indicator/SumSumKdnn/input_from_feature_columns/input_layer/marital_status_indicator/one_hotYdnn/input_from_feature_columns/input_layer/marital_status_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:?????????
?
Idnn/input_from_feature_columns/input_layer/marital_status_indicator/ShapeShapeGdnn/input_from_feature_columns/input_layer/marital_status_indicator/Sum*
T0*
_output_shapes
:
?
Wdnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Ydnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
?
Ydnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Qdnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_sliceStridedSliceIdnn/input_from_feature_columns/input_layer/marital_status_indicator/ShapeWdnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_slice/stackYdnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_slice/stack_1Ydnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
?
Sdnn/input_from_feature_columns/input_layer/marital_status_indicator/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Qdnn/input_from_feature_columns/input_layer/marital_status_indicator/Reshape/shapePackQdnn/input_from_feature_columns/input_layer/marital_status_indicator/strided_sliceSdnn/input_from_feature_columns/input_layer/marital_status_indicator/Reshape/shape/1*
T0*
N*
_output_shapes
:
?
Kdnn/input_from_feature_columns/input_layer/marital_status_indicator/ReshapeReshapeGdnn/input_from_feature_columns/input_layer/marital_status_indicator/SumQdnn/input_from_feature_columns/input_layer/marital_status_indicator/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Rdnn/input_from_feature_columns/input_layer/native_country_embedding/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Ndnn/input_from_feature_columns/input_layer/native_country_embedding/ExpandDims
ExpandDimsPlaceholder_7Rdnn/input_from_feature_columns/input_layer/native_country_embedding/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
bdnn/input_from_feature_columns/input_layer/native_country_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
\dnn/input_from_feature_columns/input_layer/native_country_embedding/to_sparse_input/NotEqualNotEqualNdnn/input_from_feature_columns/input_layer/native_country_embedding/ExpandDimsbdnn/input_from_feature_columns/input_layer/native_country_embedding/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:?????????
?
[dnn/input_from_feature_columns/input_layer/native_country_embedding/to_sparse_input/indicesWhere\dnn/input_from_feature_columns/input_layer/native_country_embedding/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Zdnn/input_from_feature_columns/input_layer/native_country_embedding/to_sparse_input/valuesGatherNdNdnn/input_from_feature_columns/input_layer/native_country_embedding/ExpandDims[dnn/input_from_feature_columns/input_layer/native_country_embedding/to_sparse_input/indices*#
_output_shapes
:?????????*
Tindices0	*
Tparams0
?
_dnn/input_from_feature_columns/input_layer/native_country_embedding/to_sparse_input/dense_shapeShapeNdnn/input_from_feature_columns/input_layer/native_country_embedding/ExpandDims*
T0*
out_type0	*
_output_shapes
:
?
Jdnn/input_from_feature_columns/input_layer/native_country_embedding/lookupStringToHashBucketFastZdnn/input_from_feature_columns/input_layer/native_country_embedding/to_sparse_input/values*
num_bucketsd*#
_output_shapes
:?????????
?
pdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
?
odnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
jdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/SliceSlice_dnn/input_from_feature_columns/input_layer/native_country_embedding/to_sparse_input/dense_shapepdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice/beginodnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
?
jdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
idnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/ProdProdjdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slicejdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Const*
T0	*
_output_shapes
: 
?
udnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
rdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
mdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GatherV2GatherV2_dnn/input_from_feature_columns/input_layer/native_country_embedding/to_sparse_input/dense_shapeudnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GatherV2/indicesrdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GatherV2/axis*
_output_shapes
: *
Taxis0*
Tindices0*
Tparams0	
?
kdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Cast/xPackidnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Prodmdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GatherV2*
N*
_output_shapes
:*
T0	
?
rdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/SparseReshapeSparseReshape[dnn/input_from_feature_columns/input_layer/native_country_embedding/to_sparse_input/indices_dnn/input_from_feature_columns/input_layer/native_country_embedding/to_sparse_input/dense_shapekdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Cast/x*-
_output_shapes
:?????????:
?
{dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/SparseReshape/IdentityIdentityJdnn/input_from_feature_columns/input_layer/native_country_embedding/lookup*
T0	*#
_output_shapes
:?????????
?
sdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
qdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GreaterEqualGreaterEqual{dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/SparseReshape/Identitysdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
jdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/WhereWhereqdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GreaterEqual*'
_output_shapes
:?????????
?
rdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
ldnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/ReshapeReshapejdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Whererdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
tdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
odnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GatherV2_1GatherV2rdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/SparseReshapeldnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Reshapetdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GatherV2_1/axis*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????*
Taxis0
?
tdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
odnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GatherV2_2GatherV2{dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/SparseReshape/Identityldnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Reshapetdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
mdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/IdentityIdentitytdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/SparseReshape:1*
T0	*
_output_shapes
:
?
~dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsodnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GatherV2_1odnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/GatherV2_2mdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Identity~dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/strided_slice/stack?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:?????????*
T0	*
Index0
?
?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/CastCast?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:?????????*

DstT0
?
?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/UniqueUnique?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGather\dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/Unique*
Tindices0	*o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0*
dtype0*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*o
_classe
caloc:@dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identity?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
|dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparseSparseSegmentMean?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/Unique:1?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:?????????
?
tdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Reshape_1/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
?
ndnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Reshape_1Reshape?dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2tdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
jdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/ShapeShape|dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
?
xdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
zdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
zdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
rdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/strided_sliceStridedSlicejdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Shapexdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/strided_slice/stackzdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/strided_slice/stack_1zdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
?
ldnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
?
jdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/stackPackldnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/stack/0rdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/strided_slice*
N*
_output_shapes
:*
T0
?
idnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/TileTilendnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Reshape_1jdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/stack*
T0
*0
_output_shapes
:??????????????????
?
odnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/zeros_like	ZerosLike|dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse*'
_output_shapes
:?????????*
T0
?
ddnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weightsSelectidnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Tileodnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/zeros_like|dnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
kdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Cast_1Cast_dnn/input_from_feature_columns/input_layer/native_country_embedding/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
?
rdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
qdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
ldnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice_1Slicekdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Cast_1rdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice_1/beginqdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
ldnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Shape_1Shapeddnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights*
T0*
_output_shapes
:
?
rdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
?
qdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice_2/sizeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
ldnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice_2Sliceldnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Shape_1rdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice_2/beginqdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
pdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
kdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/concatConcatV2ldnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice_1ldnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Slice_2pdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/concat/axis*
N*
_output_shapes
:*
T0
?
ndnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Reshape_2Reshapeddnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weightskdnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/concat*
T0*'
_output_shapes
:?????????
?
Idnn/input_from_feature_columns/input_layer/native_country_embedding/ShapeShapendnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Reshape_2*
T0*
_output_shapes
:
?
Wdnn/input_from_feature_columns/input_layer/native_country_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Ydnn/input_from_feature_columns/input_layer/native_country_embedding/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
?
Ydnn/input_from_feature_columns/input_layer/native_country_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Qdnn/input_from_feature_columns/input_layer/native_country_embedding/strided_sliceStridedSliceIdnn/input_from_feature_columns/input_layer/native_country_embedding/ShapeWdnn/input_from_feature_columns/input_layer/native_country_embedding/strided_slice/stackYdnn/input_from_feature_columns/input_layer/native_country_embedding/strided_slice/stack_1Ydnn/input_from_feature_columns/input_layer/native_country_embedding/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
?
Sdnn/input_from_feature_columns/input_layer/native_country_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Qdnn/input_from_feature_columns/input_layer/native_country_embedding/Reshape/shapePackQdnn/input_from_feature_columns/input_layer/native_country_embedding/strided_sliceSdnn/input_from_feature_columns/input_layer/native_country_embedding/Reshape/shape/1*
N*
_output_shapes
:*
T0
?
Kdnn/input_from_feature_columns/input_layer/native_country_embedding/ReshapeReshapendnn/input_from_feature_columns/input_layer/native_country_embedding/native_country_embedding_weights/Reshape_2Qdnn/input_from_feature_columns/input_layer/native_country_embedding/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Ndnn/input_from_feature_columns/input_layer/occupation_embedding/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
?????????
?
Jdnn/input_from_feature_columns/input_layer/occupation_embedding/ExpandDims
ExpandDimsPlaceholder_6Ndnn/input_from_feature_columns/input_layer/occupation_embedding/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
^dnn/input_from_feature_columns/input_layer/occupation_embedding/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
Xdnn/input_from_feature_columns/input_layer/occupation_embedding/to_sparse_input/NotEqualNotEqualJdnn/input_from_feature_columns/input_layer/occupation_embedding/ExpandDims^dnn/input_from_feature_columns/input_layer/occupation_embedding/to_sparse_input/ignore_value/x*'
_output_shapes
:?????????*
T0
?
Wdnn/input_from_feature_columns/input_layer/occupation_embedding/to_sparse_input/indicesWhereXdnn/input_from_feature_columns/input_layer/occupation_embedding/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Vdnn/input_from_feature_columns/input_layer/occupation_embedding/to_sparse_input/valuesGatherNdJdnn/input_from_feature_columns/input_layer/occupation_embedding/ExpandDimsWdnn/input_from_feature_columns/input_layer/occupation_embedding/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:?????????
?
[dnn/input_from_feature_columns/input_layer/occupation_embedding/to_sparse_input/dense_shapeShapeJdnn/input_from_feature_columns/input_layer/occupation_embedding/ExpandDims*
T0*
out_type0	*
_output_shapes
:
?
Fdnn/input_from_feature_columns/input_layer/occupation_embedding/lookupStringToHashBucketFastVdnn/input_from_feature_columns/input_layer/occupation_embedding/to_sparse_input/values*
num_bucketsd*#
_output_shapes
:?????????
?
hdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
?
gdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
bdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SliceSlice[dnn/input_from_feature_columns/input_layer/occupation_embedding/to_sparse_input/dense_shapehdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice/begingdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice/size*
Index0*
T0	*
_output_shapes
:
?
bdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
?
adnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/ProdProdbdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slicebdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Const*
T0	*
_output_shapes
: 
?
mdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
ednn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2GatherV2[dnn/input_from_feature_columns/input_layer/occupation_embedding/to_sparse_input/dense_shapemdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2/indicesjdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Taxis0*
Tindices0
?
cdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Cast/xPackadnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Prodednn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2*
T0	*
N*
_output_shapes
:
?
jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshapeSparseReshapeWdnn/input_from_feature_columns/input_layer/occupation_embedding/to_sparse_input/indices[dnn/input_from_feature_columns/input_layer/occupation_embedding/to_sparse_input/dense_shapecdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Cast/x*-
_output_shapes
:?????????:
?
sdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshape/IdentityIdentityFdnn/input_from_feature_columns/input_layer/occupation_embedding/lookup*#
_output_shapes
:?????????*
T0	
?
kdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
idnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GreaterEqualGreaterEqualsdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshape/Identitykdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
bdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/WhereWhereidnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GreaterEqual*'
_output_shapes
:?????????
?
jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
ddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/ReshapeReshapebdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Wherejdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
ldnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
gdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_1GatherV2jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshapeddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshapeldnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_1/axis*'
_output_shapes
:?????????*
Taxis0*
Tindices0	*
Tparams0	
?
ldnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
gdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_2GatherV2sdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshape/Identityddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshapeldnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_2/axis*
Taxis0*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????
?
ednn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/IdentityIdentityldnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseReshape:1*
_output_shapes
:*
T0	
?
vdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsgdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_1gdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/GatherV2_2ednn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Identityvdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/Const*T
_output_shapesB
@:?????????:?????????:?????????:?????????*
T0	
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stack?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stack_1?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice/stack_2*

begin_mask*
end_mask*#
_output_shapes
:?????????*
T0	*
Index0*
shrink_axis_mask
?
ydnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/CastCast?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:?????????*

DstT0
?
{dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/UniqueUnique?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookupResourceGatherXdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0{dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/Unique*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
dtype0*'
_output_shapes
:?????????*
Tindices0	
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup/IdentityIdentity?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup*
T0*k
_classa
_]loc:@dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1Identity?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
tdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparseSparseSegmentMean?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/embedding_lookup/Identity_1}dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/Unique:1ydnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:?????????
?
ldnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_1/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
?
fdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_1Reshape?dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/SparseFillEmptyRows/SparseFillEmptyRows:2ldnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_1/shape*'
_output_shapes
:?????????*
T0

?
bdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/ShapeShapetdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse*
T0*
_output_shapes
:
?
pdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
rdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
rdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_sliceStridedSlicebdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Shapepdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stackrdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stack_1rdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
ddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
?
bdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/stackPackddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/stack/0jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/strided_slice*
T0*
N*
_output_shapes
:
?
adnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/TileTilefdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_1bdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/stack*
T0
*0
_output_shapes
:??????????????????
?
gdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/zeros_like	ZerosLiketdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
\dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weightsSelectadnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Tilegdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/zeros_liketdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
cdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Cast_1Cast[dnn/input_from_feature_columns/input_layer/occupation_embedding/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
?
jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
idnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
ddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1Slicecdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Cast_1jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1/beginidnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
ddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Shape_1Shape\dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights*
T0*
_output_shapes
:
?
jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
?
idnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2/sizeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
ddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2Sliceddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Shape_1jdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2/beginidnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
hdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/concat/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
cdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/concatConcatV2ddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_1ddnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Slice_2hdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/concat/axis*
T0*
N*
_output_shapes
:
?
fdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_2Reshape\dnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weightscdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/concat*
T0*'
_output_shapes
:?????????
?
Ednn/input_from_feature_columns/input_layer/occupation_embedding/ShapeShapefdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_2*
_output_shapes
:*
T0
?
Sdnn/input_from_feature_columns/input_layer/occupation_embedding/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Udnn/input_from_feature_columns/input_layer/occupation_embedding/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Udnn/input_from_feature_columns/input_layer/occupation_embedding/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Mdnn/input_from_feature_columns/input_layer/occupation_embedding/strided_sliceStridedSliceEdnn/input_from_feature_columns/input_layer/occupation_embedding/ShapeSdnn/input_from_feature_columns/input_layer/occupation_embedding/strided_slice/stackUdnn/input_from_feature_columns/input_layer/occupation_embedding/strided_slice/stack_1Udnn/input_from_feature_columns/input_layer/occupation_embedding/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
?
Odnn/input_from_feature_columns/input_layer/occupation_embedding/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Mdnn/input_from_feature_columns/input_layer/occupation_embedding/Reshape/shapePackMdnn/input_from_feature_columns/input_layer/occupation_embedding/strided_sliceOdnn/input_from_feature_columns/input_layer/occupation_embedding/Reshape/shape/1*
N*
_output_shapes
:*
T0
?
Gdnn/input_from_feature_columns/input_layer/occupation_embedding/ReshapeReshapefdnn/input_from_feature_columns/input_layer/occupation_embedding/occupation_embedding_weights/Reshape_2Mdnn/input_from_feature_columns/input_layer/occupation_embedding/Reshape/shape*'
_output_shapes
:?????????*
T0
?
Hdnn/input_from_feature_columns/input_layer/race_indicator/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Ddnn/input_from_feature_columns/input_layer/race_indicator/ExpandDims
ExpandDimsPlaceholder_1Hdnn/input_from_feature_columns/input_layer/race_indicator/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
Xdnn/input_from_feature_columns/input_layer/race_indicator/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
Rdnn/input_from_feature_columns/input_layer/race_indicator/to_sparse_input/NotEqualNotEqualDdnn/input_from_feature_columns/input_layer/race_indicator/ExpandDimsXdnn/input_from_feature_columns/input_layer/race_indicator/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:?????????
?
Qdnn/input_from_feature_columns/input_layer/race_indicator/to_sparse_input/indicesWhereRdnn/input_from_feature_columns/input_layer/race_indicator/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Pdnn/input_from_feature_columns/input_layer/race_indicator/to_sparse_input/valuesGatherNdDdnn/input_from_feature_columns/input_layer/race_indicator/ExpandDimsQdnn/input_from_feature_columns/input_layer/race_indicator/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:?????????
?
Udnn/input_from_feature_columns/input_layer/race_indicator/to_sparse_input/dense_shapeShapeDdnn/input_from_feature_columns/input_layer/race_indicator/ExpandDims*
T0*
out_type0	*
_output_shapes
:
?
Kdnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/ConstConst*U
valueLBJB Amer-Indian-EskimoB Asian-Pac-IslanderB BlackB OtherB White*
dtype0*
_output_shapes
:
?
Jdnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/SizeConst*
dtype0*
_output_shapes
: *
value	B :
?
Qdnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
?
Qdnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
?
Kdnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/rangeRangeQdnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/range/startJdnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/SizeQdnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/range/delta*
_output_shapes
:
?
Jdnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/CastCastKdnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/range*
_output_shapes
:*

DstT0	*

SrcT0
?
Vdnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/hash_table/ConstConst*
dtype0	*
_output_shapes
: *
valueB	 R
?????????
?
[dnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/hash_table/hash_tableHashTableV2*
	key_dtype0*
value_dtype0	*
_output_shapes
: 
?
odnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2[dnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/hash_table/hash_tableKdnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/ConstJdnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/Cast*	
Tin0*

Tout0	
?
]dnn/input_from_feature_columns/input_layer/race_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2[dnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/hash_table/hash_tablePdnn/input_from_feature_columns/input_layer/race_indicator/to_sparse_input/valuesVdnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/hash_table/Const*

Tout0	*#
_output_shapes
:?????????*	
Tin0
?
Udnn/input_from_feature_columns/input_layer/race_indicator/SparseToDense/default_valueConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
Gdnn/input_from_feature_columns/input_layer/race_indicator/SparseToDenseSparseToDenseQdnn/input_from_feature_columns/input_layer/race_indicator/to_sparse_input/indicesUdnn/input_from_feature_columns/input_layer/race_indicator/to_sparse_input/dense_shape]dnn/input_from_feature_columns/input_layer/race_indicator/hash_table_Lookup/LookupTableFindV2Udnn/input_from_feature_columns/input_layer/race_indicator/SparseToDense/default_value*'
_output_shapes
:?????????*
Tindices0	*
T0	
?
Gdnn/input_from_feature_columns/input_layer/race_indicator/one_hot/ConstConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Idnn/input_from_feature_columns/input_layer/race_indicator/one_hot/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Gdnn/input_from_feature_columns/input_layer/race_indicator/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
?
Jdnn/input_from_feature_columns/input_layer/race_indicator/one_hot/on_valueConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Kdnn/input_from_feature_columns/input_layer/race_indicator/one_hot/off_valueConst*
dtype0*
_output_shapes
: *
valueB
 *    
?
Adnn/input_from_feature_columns/input_layer/race_indicator/one_hotOneHotGdnn/input_from_feature_columns/input_layer/race_indicator/SparseToDenseGdnn/input_from_feature_columns/input_layer/race_indicator/one_hot/depthJdnn/input_from_feature_columns/input_layer/race_indicator/one_hot/on_valueKdnn/input_from_feature_columns/input_layer/race_indicator/one_hot/off_value*
T0*+
_output_shapes
:?????????
?
Odnn/input_from_feature_columns/input_layer/race_indicator/Sum/reduction_indicesConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
=dnn/input_from_feature_columns/input_layer/race_indicator/SumSumAdnn/input_from_feature_columns/input_layer/race_indicator/one_hotOdnn/input_from_feature_columns/input_layer/race_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:?????????
?
?dnn/input_from_feature_columns/input_layer/race_indicator/ShapeShape=dnn/input_from_feature_columns/input_layer/race_indicator/Sum*
T0*
_output_shapes
:
?
Mdnn/input_from_feature_columns/input_layer/race_indicator/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Odnn/input_from_feature_columns/input_layer/race_indicator/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Odnn/input_from_feature_columns/input_layer/race_indicator/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Gdnn/input_from_feature_columns/input_layer/race_indicator/strided_sliceStridedSlice?dnn/input_from_feature_columns/input_layer/race_indicator/ShapeMdnn/input_from_feature_columns/input_layer/race_indicator/strided_slice/stackOdnn/input_from_feature_columns/input_layer/race_indicator/strided_slice/stack_1Odnn/input_from_feature_columns/input_layer/race_indicator/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
?
Idnn/input_from_feature_columns/input_layer/race_indicator/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Gdnn/input_from_feature_columns/input_layer/race_indicator/Reshape/shapePackGdnn/input_from_feature_columns/input_layer/race_indicator/strided_sliceIdnn/input_from_feature_columns/input_layer/race_indicator/Reshape/shape/1*
T0*
N*
_output_shapes
:
?
Adnn/input_from_feature_columns/input_layer/race_indicator/ReshapeReshape=dnn/input_from_feature_columns/input_layer/race_indicator/SumGdnn/input_from_feature_columns/input_layer/race_indicator/Reshape/shape*'
_output_shapes
:?????????*
T0
?
Pdnn/input_from_feature_columns/input_layer/relationship_indicator/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Ldnn/input_from_feature_columns/input_layer/relationship_indicator/ExpandDims
ExpandDimsPlaceholder_4Pdnn/input_from_feature_columns/input_layer/relationship_indicator/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
`dnn/input_from_feature_columns/input_layer/relationship_indicator/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
Zdnn/input_from_feature_columns/input_layer/relationship_indicator/to_sparse_input/NotEqualNotEqualLdnn/input_from_feature_columns/input_layer/relationship_indicator/ExpandDims`dnn/input_from_feature_columns/input_layer/relationship_indicator/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:?????????
?
Ydnn/input_from_feature_columns/input_layer/relationship_indicator/to_sparse_input/indicesWhereZdnn/input_from_feature_columns/input_layer/relationship_indicator/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Xdnn/input_from_feature_columns/input_layer/relationship_indicator/to_sparse_input/valuesGatherNdLdnn/input_from_feature_columns/input_layer/relationship_indicator/ExpandDimsYdnn/input_from_feature_columns/input_layer/relationship_indicator/to_sparse_input/indices*#
_output_shapes
:?????????*
Tindices0	*
Tparams0
?
]dnn/input_from_feature_columns/input_layer/relationship_indicator/to_sparse_input/dense_shapeShapeLdnn/input_from_feature_columns/input_layer/relationship_indicator/ExpandDims*
T0*
out_type0	*
_output_shapes
:
?
[dnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/ConstConst*]
valueTBRB HusbandB Not-in-familyB WifeB
 Own-childB
 UnmarriedB Other-relative*
dtype0*
_output_shapes
:
?
Zdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/SizeConst*
dtype0*
_output_shapes
: *
value	B :
?
adnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
?
adnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
[dnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/rangeRangeadnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/range/startZdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/Sizeadnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/range/delta*
_output_shapes
:
?
Zdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/CastCast[dnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/range*

SrcT0*
_output_shapes
:*

DstT0	
?
fdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/ConstConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
kdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/hash_tableHashTableV2*
value_dtype0	*
	key_dtype0*
_output_shapes
: 
?
dnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2kdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/hash_table[dnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/ConstZdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/Cast*	
Tin0*

Tout0	
?
ednn/input_from_feature_columns/input_layer/relationship_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2kdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/hash_tableXdnn/input_from_feature_columns/input_layer/relationship_indicator/to_sparse_input/valuesfdnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/Const*#
_output_shapes
:?????????*	
Tin0*

Tout0	
?
]dnn/input_from_feature_columns/input_layer/relationship_indicator/SparseToDense/default_valueConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
Odnn/input_from_feature_columns/input_layer/relationship_indicator/SparseToDenseSparseToDenseYdnn/input_from_feature_columns/input_layer/relationship_indicator/to_sparse_input/indices]dnn/input_from_feature_columns/input_layer/relationship_indicator/to_sparse_input/dense_shapeednn/input_from_feature_columns/input_layer/relationship_indicator/hash_table_Lookup/LookupTableFindV2]dnn/input_from_feature_columns/input_layer/relationship_indicator/SparseToDense/default_value*'
_output_shapes
:?????????*
Tindices0	*
T0	
?
Odnn/input_from_feature_columns/input_layer/relationship_indicator/one_hot/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *  ??
?
Qdnn/input_from_feature_columns/input_layer/relationship_indicator/one_hot/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Odnn/input_from_feature_columns/input_layer/relationship_indicator/one_hot/depthConst*
value	B :*
dtype0*
_output_shapes
: 
?
Rdnn/input_from_feature_columns/input_layer/relationship_indicator/one_hot/on_valueConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Sdnn/input_from_feature_columns/input_layer/relationship_indicator/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Idnn/input_from_feature_columns/input_layer/relationship_indicator/one_hotOneHotOdnn/input_from_feature_columns/input_layer/relationship_indicator/SparseToDenseOdnn/input_from_feature_columns/input_layer/relationship_indicator/one_hot/depthRdnn/input_from_feature_columns/input_layer/relationship_indicator/one_hot/on_valueSdnn/input_from_feature_columns/input_layer/relationship_indicator/one_hot/off_value*
T0*+
_output_shapes
:?????????
?
Wdnn/input_from_feature_columns/input_layer/relationship_indicator/Sum/reduction_indicesConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Ednn/input_from_feature_columns/input_layer/relationship_indicator/SumSumIdnn/input_from_feature_columns/input_layer/relationship_indicator/one_hotWdnn/input_from_feature_columns/input_layer/relationship_indicator/Sum/reduction_indices*
T0*'
_output_shapes
:?????????
?
Gdnn/input_from_feature_columns/input_layer/relationship_indicator/ShapeShapeEdnn/input_from_feature_columns/input_layer/relationship_indicator/Sum*
T0*
_output_shapes
:
?
Udnn/input_from_feature_columns/input_layer/relationship_indicator/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Wdnn/input_from_feature_columns/input_layer/relationship_indicator/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Wdnn/input_from_feature_columns/input_layer/relationship_indicator/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Odnn/input_from_feature_columns/input_layer/relationship_indicator/strided_sliceStridedSliceGdnn/input_from_feature_columns/input_layer/relationship_indicator/ShapeUdnn/input_from_feature_columns/input_layer/relationship_indicator/strided_slice/stackWdnn/input_from_feature_columns/input_layer/relationship_indicator/strided_slice/stack_1Wdnn/input_from_feature_columns/input_layer/relationship_indicator/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
?
Qdnn/input_from_feature_columns/input_layer/relationship_indicator/Reshape/shape/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Odnn/input_from_feature_columns/input_layer/relationship_indicator/Reshape/shapePackOdnn/input_from_feature_columns/input_layer/relationship_indicator/strided_sliceQdnn/input_from_feature_columns/input_layer/relationship_indicator/Reshape/shape/1*
T0*
N*
_output_shapes
:
?
Idnn/input_from_feature_columns/input_layer/relationship_indicator/ReshapeReshapeEdnn/input_from_feature_columns/input_layer/relationship_indicator/SumOdnn/input_from_feature_columns/input_layer/relationship_indicator/Reshape/shape*
T0*'
_output_shapes
:?????????
?
Mdnn/input_from_feature_columns/input_layer/workclass_indicator/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
?????????
?
Idnn/input_from_feature_columns/input_layer/workclass_indicator/ExpandDims
ExpandDimsPlaceholder_5Mdnn/input_from_feature_columns/input_layer/workclass_indicator/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
]dnn/input_from_feature_columns/input_layer/workclass_indicator/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
Wdnn/input_from_feature_columns/input_layer/workclass_indicator/to_sparse_input/NotEqualNotEqualIdnn/input_from_feature_columns/input_layer/workclass_indicator/ExpandDims]dnn/input_from_feature_columns/input_layer/workclass_indicator/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:?????????
?
Vdnn/input_from_feature_columns/input_layer/workclass_indicator/to_sparse_input/indicesWhereWdnn/input_from_feature_columns/input_layer/workclass_indicator/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Udnn/input_from_feature_columns/input_layer/workclass_indicator/to_sparse_input/valuesGatherNdIdnn/input_from_feature_columns/input_layer/workclass_indicator/ExpandDimsVdnn/input_from_feature_columns/input_layer/workclass_indicator/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:?????????
?
Zdnn/input_from_feature_columns/input_layer/workclass_indicator/to_sparse_input/dense_shapeShapeIdnn/input_from_feature_columns/input_layer/workclass_indicator/ExpandDims*
T0*
out_type0	*
_output_shapes
:
?
Udnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/ConstConst*?
value}B{	B Self-emp-not-incB PrivateB
 State-govB Federal-govB
 Local-govB ?B Self-emp-incB Without-payB Never-worked*
dtype0*
_output_shapes
:	
?
Tdnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/SizeConst*
value	B :	*
dtype0*
_output_shapes
: 
?
[dnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
?
[dnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
Udnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/rangeRange[dnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/range/startTdnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/Size[dnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/range/delta*
_output_shapes
:	
?
Tdnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/CastCastUdnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/range*
_output_shapes
:	*

DstT0	*

SrcT0
?
`dnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/hash_table/ConstConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
ednn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/hash_table/hash_tableHashTableV2*
	key_dtype0*
value_dtype0	*
_output_shapes
: 
?
ydnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2ednn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/hash_table/hash_tableUdnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/ConstTdnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/Cast*

Tout0	*	
Tin0
?
bdnn/input_from_feature_columns/input_layer/workclass_indicator/hash_table_Lookup/LookupTableFindV2LookupTableFindV2ednn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/hash_table/hash_tableUdnn/input_from_feature_columns/input_layer/workclass_indicator/to_sparse_input/values`dnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/hash_table/Const*#
_output_shapes
:?????????*	
Tin0*

Tout0	
?
Zdnn/input_from_feature_columns/input_layer/workclass_indicator/SparseToDense/default_valueConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
Ldnn/input_from_feature_columns/input_layer/workclass_indicator/SparseToDenseSparseToDenseVdnn/input_from_feature_columns/input_layer/workclass_indicator/to_sparse_input/indicesZdnn/input_from_feature_columns/input_layer/workclass_indicator/to_sparse_input/dense_shapebdnn/input_from_feature_columns/input_layer/workclass_indicator/hash_table_Lookup/LookupTableFindV2Zdnn/input_from_feature_columns/input_layer/workclass_indicator/SparseToDense/default_value*'
_output_shapes
:?????????*
Tindices0	*
T0	
?
Ldnn/input_from_feature_columns/input_layer/workclass_indicator/one_hot/ConstConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Ndnn/input_from_feature_columns/input_layer/workclass_indicator/one_hot/Const_1Const*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ldnn/input_from_feature_columns/input_layer/workclass_indicator/one_hot/depthConst*
value	B :	*
dtype0*
_output_shapes
: 
?
Odnn/input_from_feature_columns/input_layer/workclass_indicator/one_hot/on_valueConst*
valueB
 *  ??*
dtype0*
_output_shapes
: 
?
Pdnn/input_from_feature_columns/input_layer/workclass_indicator/one_hot/off_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Fdnn/input_from_feature_columns/input_layer/workclass_indicator/one_hotOneHotLdnn/input_from_feature_columns/input_layer/workclass_indicator/SparseToDenseLdnn/input_from_feature_columns/input_layer/workclass_indicator/one_hot/depthOdnn/input_from_feature_columns/input_layer/workclass_indicator/one_hot/on_valuePdnn/input_from_feature_columns/input_layer/workclass_indicator/one_hot/off_value*
T0*+
_output_shapes
:?????????	
?
Tdnn/input_from_feature_columns/input_layer/workclass_indicator/Sum/reduction_indicesConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Bdnn/input_from_feature_columns/input_layer/workclass_indicator/SumSumFdnn/input_from_feature_columns/input_layer/workclass_indicator/one_hotTdnn/input_from_feature_columns/input_layer/workclass_indicator/Sum/reduction_indices*'
_output_shapes
:?????????	*
T0
?
Ddnn/input_from_feature_columns/input_layer/workclass_indicator/ShapeShapeBdnn/input_from_feature_columns/input_layer/workclass_indicator/Sum*
T0*
_output_shapes
:
?
Rdnn/input_from_feature_columns/input_layer/workclass_indicator/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
?
Tdnn/input_from_feature_columns/input_layer/workclass_indicator/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Tdnn/input_from_feature_columns/input_layer/workclass_indicator/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Ldnn/input_from_feature_columns/input_layer/workclass_indicator/strided_sliceStridedSliceDdnn/input_from_feature_columns/input_layer/workclass_indicator/ShapeRdnn/input_from_feature_columns/input_layer/workclass_indicator/strided_slice/stackTdnn/input_from_feature_columns/input_layer/workclass_indicator/strided_slice/stack_1Tdnn/input_from_feature_columns/input_layer/workclass_indicator/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
?
Ndnn/input_from_feature_columns/input_layer/workclass_indicator/Reshape/shape/1Const*
dtype0*
_output_shapes
: *
value	B :	
?
Ldnn/input_from_feature_columns/input_layer/workclass_indicator/Reshape/shapePackLdnn/input_from_feature_columns/input_layer/workclass_indicator/strided_sliceNdnn/input_from_feature_columns/input_layer/workclass_indicator/Reshape/shape/1*
N*
_output_shapes
:*
T0
?
Fdnn/input_from_feature_columns/input_layer/workclass_indicator/ReshapeReshapeBdnn/input_from_feature_columns/input_layer/workclass_indicator/SumLdnn/input_from_feature_columns/input_layer/workclass_indicator/Reshape/shape*
T0*'
_output_shapes
:?????????	
?
6dnn/input_from_feature_columns/input_layer/concat/axisConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
1dnn/input_from_feature_columns/input_layer/concatConcatV26dnn/input_from_feature_columns/input_layer/age/Reshape?dnn/input_from_feature_columns/input_layer/capital_gain/Reshape?dnn/input_from_feature_columns/input_layer/capital_loss/ReshapeFdnn/input_from_feature_columns/input_layer/education_indicator/Reshape@dnn/input_from_feature_columns/input_layer/education_num/ReshapeCdnn/input_from_feature_columns/input_layer/gender_indicator/ReshapeAdnn/input_from_feature_columns/input_layer/hours_per_week/ReshapeKdnn/input_from_feature_columns/input_layer/marital_status_indicator/ReshapeKdnn/input_from_feature_columns/input_layer/native_country_embedding/ReshapeGdnn/input_from_feature_columns/input_layer/occupation_embedding/ReshapeAdnn/input_from_feature_columns/input_layer/race_indicator/ReshapeIdnn/input_from_feature_columns/input_layer/relationship_indicator/ReshapeFdnn/input_from_feature_columns/input_layer/workclass_indicator/Reshape6dnn/input_from_feature_columns/input_layer/concat/axis*
T0*
N*'
_output_shapes
:?????????B
?
@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"B   d   *2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
?
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *?B?*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
?
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *?B>*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
?
Hdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:Bd
?
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
?
>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:Bd
?
:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
_output_shapes

:Bd
?
dnn/hiddenlayer_0/kernel/part_0VarHandleOp*
shape
:Bd*0
shared_name!dnn/hiddenlayer_0/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes
: 
?
@dnn/hiddenlayer_0/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/kernel/part_0*
_output_shapes
: 
?
&dnn/hiddenlayer_0/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/kernel/part_0:dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0*
dtype0
?
3dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:Bd*2
_class(
&$loc:@dnn/hiddenlayer_0/kernel/part_0
?
/dnn/hiddenlayer_0/bias/part_0/Initializer/zerosConst*
dtype0*
_output_shapes
:d*
valueBd*    *0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
?
dnn/hiddenlayer_0/bias/part_0VarHandleOp*.
shared_namednn/hiddenlayer_0/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
: *
shape:d
?
>dnn/hiddenlayer_0/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_0/bias/part_0*
_output_shapes
: 
?
$dnn/hiddenlayer_0/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_0/bias/part_0/dnn/hiddenlayer_0/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0*
dtype0
?
1dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:d*0
_class&
$"loc:@dnn/hiddenlayer_0/bias/part_0
?
'dnn/hiddenlayer_0/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:Bd
v
dnn/hiddenlayer_0/kernelIdentity'dnn/hiddenlayer_0/kernel/ReadVariableOp*
_output_shapes

:Bd*
T0
?
dnn/hiddenlayer_0/MatMulMatMul1dnn/input_from_feature_columns/input_layer/concatdnn/hiddenlayer_0/kernel*
T0*'
_output_shapes
:?????????d

%dnn/hiddenlayer_0/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:d
n
dnn/hiddenlayer_0/biasIdentity%dnn/hiddenlayer_0/bias/ReadVariableOp*
_output_shapes
:d*
T0
?
dnn/hiddenlayer_0/BiasAddBiasAdddnn/hiddenlayer_0/MatMuldnn/hiddenlayer_0/bias*
T0*'
_output_shapes
:?????????d
k
dnn/hiddenlayer_0/ReluReludnn/hiddenlayer_0/BiasAdd*
T0*'
_output_shapes
:?????????d
g
dnn/zero_fraction/SizeSizednn/hiddenlayer_0/Relu*
T0*
out_type0	*
_output_shapes
: 
c
dnn/zero_fraction/LessEqual/yConst*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
dnn/zero_fraction/LessEqual	LessEqualdnn/zero_fraction/Sizednn/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction/cond/SwitchSwitchdnn/zero_fraction/LessEqualdnn/zero_fraction/LessEqual*
_output_shapes
: : *
T0

m
dnn/zero_fraction/cond/switch_tIdentitydnn/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
k
dnn/zero_fraction/cond/switch_fIdentitydnn/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
h
dnn/zero_fraction/cond/pred_idIdentitydnn/zero_fraction/LessEqual*
_output_shapes
: *
T0

?
*dnn/zero_fraction/cond/count_nonzero/zerosConst ^dnn/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
-dnn/zero_fraction/cond/count_nonzero/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1*dnn/zero_fraction/cond/count_nonzero/zeros*'
_output_shapes
:?????????d*
T0
?
4dnn/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_0/Reludnn/zero_fraction/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_0/Relu*:
_output_shapes(
&:?????????d:?????????d
?
)dnn/zero_fraction/cond/count_nonzero/CastCast-dnn/zero_fraction/cond/count_nonzero/NotEqual*'
_output_shapes
:?????????d*

DstT0*

SrcT0

?
*dnn/zero_fraction/cond/count_nonzero/ConstConst ^dnn/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
?
2dnn/zero_fraction/cond/count_nonzero/nonzero_countSum)dnn/zero_fraction/cond/count_nonzero/Cast*dnn/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
?
dnn/zero_fraction/cond/CastCast2dnn/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
?
,dnn/zero_fraction/cond/count_nonzero_1/zerosConst ^dnn/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
?
/dnn/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch,dnn/zero_fraction/cond/count_nonzero_1/zeros*'
_output_shapes
:?????????d*
T0
?
6dnn/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_0/Reludnn/zero_fraction/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_0/Relu*:
_output_shapes(
&:?????????d:?????????d
?
+dnn/zero_fraction/cond/count_nonzero_1/CastCast/dnn/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*'
_output_shapes
:?????????d*

DstT0	
?
,dnn/zero_fraction/cond/count_nonzero_1/ConstConst ^dnn/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
:*
valueB"       
?
4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countSum+dnn/zero_fraction/cond/count_nonzero_1/Cast,dnn/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
dnn/zero_fraction/cond/MergeMerge4dnn/zero_fraction/cond/count_nonzero_1/nonzero_countdnn/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
(dnn/zero_fraction/counts_to_fraction/subSubdnn/zero_fraction/Sizednn/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
)dnn/zero_fraction/counts_to_fraction/CastCast(dnn/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
{
+dnn/zero_fraction/counts_to_fraction/Cast_1Castdnn/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	
?
,dnn/zero_fraction/counts_to_fraction/truedivRealDiv)dnn/zero_fraction/counts_to_fraction/Cast+dnn/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
u
dnn/zero_fraction/fractionIdentity,dnn/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
?
2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_0/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
-dnn/dnn/hiddenlayer_0/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_0/fraction_of_zero_values/tagsdnn/zero_fraction/fraction*
_output_shapes
: *
T0
?
$dnn/dnn/hiddenlayer_0/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_0/activation*
dtype0*
_output_shapes
: 
?
 dnn/dnn/hiddenlayer_0/activationHistogramSummary$dnn/dnn/hiddenlayer_0/activation/tagdnn/hiddenlayer_0/Relu*
_output_shapes
: 
?
@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"d   F   *2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0
?
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *H`@?*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
?
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *H`@>*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: 
?
Hdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:dF
?
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
?
>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:dF
?
:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
_output_shapes

:dF
?
dnn/hiddenlayer_1/kernel/part_0VarHandleOp*0
shared_name!dnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes
: *
shape
:dF
?
@dnn/hiddenlayer_1/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/kernel/part_0*
_output_shapes
: 
?
&dnn/hiddenlayer_1/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/kernel/part_0:dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0
?
3dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:dF
?
/dnn/hiddenlayer_1/bias/part_0/Initializer/zerosConst*
dtype0*
_output_shapes
:F*
valueBF*    *0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0
?
dnn/hiddenlayer_1/bias/part_0VarHandleOp*
shape:F*.
shared_namednn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
: 
?
>dnn/hiddenlayer_1/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_1/bias/part_0*
_output_shapes
: 
?
$dnn/hiddenlayer_1/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_1/bias/part_0/dnn/hiddenlayer_1/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0
?
1dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:F
?
'dnn/hiddenlayer_1/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:dF
v
dnn/hiddenlayer_1/kernelIdentity'dnn/hiddenlayer_1/kernel/ReadVariableOp*
T0*
_output_shapes

:dF
?
dnn/hiddenlayer_1/MatMulMatMuldnn/hiddenlayer_0/Reludnn/hiddenlayer_1/kernel*
T0*'
_output_shapes
:?????????F

%dnn/hiddenlayer_1/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:F
n
dnn/hiddenlayer_1/biasIdentity%dnn/hiddenlayer_1/bias/ReadVariableOp*
T0*
_output_shapes
:F
?
dnn/hiddenlayer_1/BiasAddBiasAdddnn/hiddenlayer_1/MatMuldnn/hiddenlayer_1/bias*
T0*'
_output_shapes
:?????????F
k
dnn/hiddenlayer_1/ReluReludnn/hiddenlayer_1/BiasAdd*
T0*'
_output_shapes
:?????????F
i
dnn/zero_fraction_1/SizeSizednn/hiddenlayer_1/Relu*
_output_shapes
: *
T0*
out_type0	
e
dnn/zero_fraction_1/LessEqual/yConst*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
dnn/zero_fraction_1/LessEqual	LessEqualdnn/zero_fraction_1/Sizednn/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_1/cond/SwitchSwitchdnn/zero_fraction_1/LessEqualdnn/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_1/cond/switch_tIdentity!dnn/zero_fraction_1/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_1/cond/switch_fIdentitydnn/zero_fraction_1/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_1/cond/pred_idIdentitydnn/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: 
?
,dnn/zero_fraction_1/cond/count_nonzero/zerosConst"^dnn/zero_fraction_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
/dnn/zero_fraction_1/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_1/cond/count_nonzero/zeros*
T0*'
_output_shapes
:?????????F
?
6dnn/zero_fraction_1/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_1/Relu dnn/zero_fraction_1/cond/pred_id*:
_output_shapes(
&:?????????F:?????????F*
T0*)
_class
loc:@dnn/hiddenlayer_1/Relu
?
+dnn/zero_fraction_1/cond/count_nonzero/CastCast/dnn/zero_fraction_1/cond/count_nonzero/NotEqual*

SrcT0
*'
_output_shapes
:?????????F*

DstT0
?
,dnn/zero_fraction_1/cond/count_nonzero/ConstConst"^dnn/zero_fraction_1/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
4dnn/zero_fraction_1/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_1/cond/count_nonzero/Cast,dnn/zero_fraction_1/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
dnn/zero_fraction_1/cond/CastCast4dnn/zero_fraction_1/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
?
.dnn/zero_fraction_1/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_1/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_1/cond/count_nonzero_1/zeros*'
_output_shapes
:?????????F*
T0
?
8dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_1/Relu dnn/zero_fraction_1/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_1/Relu*:
_output_shapes(
&:?????????F:?????????F
?
-dnn/zero_fraction_1/cond/count_nonzero_1/CastCast1dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual*

SrcT0
*'
_output_shapes
:?????????F*

DstT0	
?
.dnn/zero_fraction_1/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_1/cond/switch_f*
dtype0*
_output_shapes
:*
valueB"       
?
6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_1/cond/count_nonzero_1/Cast.dnn/zero_fraction_1/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_1/cond/MergeMerge6dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_1/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
*dnn/zero_fraction_1/counts_to_fraction/subSubdnn/zero_fraction_1/Sizednn/zero_fraction_1/cond/Merge*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_1/counts_to_fraction/CastCast*dnn/zero_fraction_1/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	

-dnn/zero_fraction_1/counts_to_fraction/Cast_1Castdnn/zero_fraction_1/Size*

SrcT0	*
_output_shapes
: *

DstT0
?
.dnn/zero_fraction_1/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_1/counts_to_fraction/Cast-dnn/zero_fraction_1/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_1/fractionIdentity.dnn/zero_fraction_1/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_1/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
-dnn/dnn/hiddenlayer_1/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_1/fraction_of_zero_values/tagsdnn/zero_fraction_1/fraction*
_output_shapes
: *
T0
?
$dnn/dnn/hiddenlayer_1/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_1/activation*
dtype0*
_output_shapes
: 
?
 dnn/dnn/hiddenlayer_1/activationHistogramSummary$dnn/dnn/hiddenlayer_1/activation/tagdnn/hiddenlayer_1/Relu*
_output_shapes
: 
?
@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"F   0   *2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
:
?
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *??f?*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
: 
?
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *??f>*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes
: 
?
Hdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:F0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0
?
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: 
?
>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/sub*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
_output_shapes

:F0
?
:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform/min*
_output_shapes

:F0*
T0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0
?
dnn/hiddenlayer_2/kernel/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape
:F0*0
shared_name!dnn/hiddenlayer_2/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0
?
@dnn/hiddenlayer_2/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_2/kernel/part_0*
_output_shapes
: 
?
&dnn/hiddenlayer_2/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_2/kernel/part_0:dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0
?
3dnn/hiddenlayer_2/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes

:F0
?
/dnn/hiddenlayer_2/bias/part_0/Initializer/zerosConst*
dtype0*
_output_shapes
:0*
valueB0*    *0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0
?
dnn/hiddenlayer_2/bias/part_0VarHandleOp*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
dtype0*
_output_shapes
: *
shape:0*.
shared_namednn/hiddenlayer_2/bias/part_0
?
>dnn/hiddenlayer_2/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_2/bias/part_0*
_output_shapes
: 
?
$dnn/hiddenlayer_2/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_2/bias/part_0/dnn/hiddenlayer_2/bias/part_0/Initializer/zeros*
dtype0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0
?
1dnn/hiddenlayer_2/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_2/bias/part_0*
dtype0*
_output_shapes
:0
?
'dnn/hiddenlayer_2/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes

:F0
v
dnn/hiddenlayer_2/kernelIdentity'dnn/hiddenlayer_2/kernel/ReadVariableOp*
T0*
_output_shapes

:F0
?
dnn/hiddenlayer_2/MatMulMatMuldnn/hiddenlayer_1/Reludnn/hiddenlayer_2/kernel*'
_output_shapes
:?????????0*
T0

%dnn/hiddenlayer_2/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/bias/part_0*
dtype0*
_output_shapes
:0
n
dnn/hiddenlayer_2/biasIdentity%dnn/hiddenlayer_2/bias/ReadVariableOp*
T0*
_output_shapes
:0
?
dnn/hiddenlayer_2/BiasAddBiasAdddnn/hiddenlayer_2/MatMuldnn/hiddenlayer_2/bias*
T0*'
_output_shapes
:?????????0
k
dnn/hiddenlayer_2/ReluReludnn/hiddenlayer_2/BiasAdd*'
_output_shapes
:?????????0*
T0
i
dnn/zero_fraction_2/SizeSizednn/hiddenlayer_2/Relu*
T0*
out_type0	*
_output_shapes
: 
e
dnn/zero_fraction_2/LessEqual/yConst*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
dnn/zero_fraction_2/LessEqual	LessEqualdnn/zero_fraction_2/Sizednn/zero_fraction_2/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_2/cond/SwitchSwitchdnn/zero_fraction_2/LessEqualdnn/zero_fraction_2/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_2/cond/switch_tIdentity!dnn/zero_fraction_2/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_2/cond/switch_fIdentitydnn/zero_fraction_2/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_2/cond/pred_idIdentitydnn/zero_fraction_2/LessEqual*
T0
*
_output_shapes
: 
?
,dnn/zero_fraction_2/cond/count_nonzero/zerosConst"^dnn/zero_fraction_2/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
/dnn/zero_fraction_2/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_2/cond/count_nonzero/zeros*
T0*'
_output_shapes
:?????????0
?
6dnn/zero_fraction_2/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_2/Relu dnn/zero_fraction_2/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_2/Relu*:
_output_shapes(
&:?????????0:?????????0
?
+dnn/zero_fraction_2/cond/count_nonzero/CastCast/dnn/zero_fraction_2/cond/count_nonzero/NotEqual*

SrcT0
*'
_output_shapes
:?????????0*

DstT0
?
,dnn/zero_fraction_2/cond/count_nonzero/ConstConst"^dnn/zero_fraction_2/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
?
4dnn/zero_fraction_2/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_2/cond/count_nonzero/Cast,dnn/zero_fraction_2/cond/count_nonzero/Const*
_output_shapes
: *
T0
?
dnn/zero_fraction_2/cond/CastCast4dnn/zero_fraction_2/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
?
.dnn/zero_fraction_2/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_2/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_2/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:?????????0
?
8dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_2/Relu dnn/zero_fraction_2/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_2/Relu*:
_output_shapes(
&:?????????0:?????????0
?
-dnn/zero_fraction_2/cond/count_nonzero_1/CastCast1dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual*

SrcT0
*'
_output_shapes
:?????????0*

DstT0	
?
.dnn/zero_fraction_2/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_2/cond/switch_f*
dtype0*
_output_shapes
:*
valueB"       
?
6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_2/cond/count_nonzero_1/Cast.dnn/zero_fraction_2/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_2/cond/MergeMerge6dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_2/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
*dnn/zero_fraction_2/counts_to_fraction/subSubdnn/zero_fraction_2/Sizednn/zero_fraction_2/cond/Merge*
T0	*
_output_shapes
: 
?
+dnn/zero_fraction_2/counts_to_fraction/CastCast*dnn/zero_fraction_2/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	

-dnn/zero_fraction_2/counts_to_fraction/Cast_1Castdnn/zero_fraction_2/Size*

SrcT0	*
_output_shapes
: *

DstT0
?
.dnn/zero_fraction_2/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_2/counts_to_fraction/Cast-dnn/zero_fraction_2/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_2/fractionIdentity.dnn/zero_fraction_2/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_2/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
-dnn/dnn/hiddenlayer_2/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_2/fraction_of_zero_values/tagsdnn/zero_fraction_2/fraction*
_output_shapes
: *
T0
?
$dnn/dnn/hiddenlayer_2/activation/tagConst*1
value(B& B dnn/dnn/hiddenlayer_2/activation*
dtype0*
_output_shapes
: 
?
 dnn/dnn/hiddenlayer_2/activationHistogramSummary$dnn/dnn/hiddenlayer_2/activation/tagdnn/hiddenlayer_2/Relu*
_output_shapes
: 
?
@dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/shapeConst*
valueB"0   "   *2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes
:
?
>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *??*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes
: 
?
>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *?>*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes
: 
?
Hdnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform@dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/shape*
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes

:0"
?
>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/subSub>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/max>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
_output_shapes
: 
?
>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/mulMulHdnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/RandomUniform>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/sub*
_output_shapes

:0"*
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0
?
:dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniformAdd>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/mul>dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform/min*
T0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
_output_shapes

:0"
?
dnn/hiddenlayer_3/kernel/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape
:0"*0
shared_name!dnn/hiddenlayer_3/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0
?
@dnn/hiddenlayer_3/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_3/kernel/part_0*
_output_shapes
: 
?
&dnn/hiddenlayer_3/kernel/part_0/AssignAssignVariableOpdnn/hiddenlayer_3/kernel/part_0:dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
dtype0
?
3dnn/hiddenlayer_3/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/kernel/part_0*2
_class(
&$loc:@dnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes

:0"
?
/dnn/hiddenlayer_3/bias/part_0/Initializer/zerosConst*
dtype0*
_output_shapes
:"*
valueB"*    *0
_class&
$"loc:@dnn/hiddenlayer_3/bias/part_0
?
dnn/hiddenlayer_3/bias/part_0VarHandleOp*.
shared_namednn/hiddenlayer_3/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_3/bias/part_0*
dtype0*
_output_shapes
: *
shape:"
?
>dnn/hiddenlayer_3/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/hiddenlayer_3/bias/part_0*
_output_shapes
: 
?
$dnn/hiddenlayer_3/bias/part_0/AssignAssignVariableOpdnn/hiddenlayer_3/bias/part_0/dnn/hiddenlayer_3/bias/part_0/Initializer/zeros*0
_class&
$"loc:@dnn/hiddenlayer_3/bias/part_0*
dtype0
?
1dnn/hiddenlayer_3/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/bias/part_0*0
_class&
$"loc:@dnn/hiddenlayer_3/bias/part_0*
dtype0*
_output_shapes
:"
?
'dnn/hiddenlayer_3/kernel/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes

:0"
v
dnn/hiddenlayer_3/kernelIdentity'dnn/hiddenlayer_3/kernel/ReadVariableOp*
T0*
_output_shapes

:0"
?
dnn/hiddenlayer_3/MatMulMatMuldnn/hiddenlayer_2/Reludnn/hiddenlayer_3/kernel*
T0*'
_output_shapes
:?????????"

%dnn/hiddenlayer_3/bias/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/bias/part_0*
dtype0*
_output_shapes
:"
n
dnn/hiddenlayer_3/biasIdentity%dnn/hiddenlayer_3/bias/ReadVariableOp*
_output_shapes
:"*
T0
?
dnn/hiddenlayer_3/BiasAddBiasAdddnn/hiddenlayer_3/MatMuldnn/hiddenlayer_3/bias*
T0*'
_output_shapes
:?????????"
k
dnn/hiddenlayer_3/ReluReludnn/hiddenlayer_3/BiasAdd*
T0*'
_output_shapes
:?????????"
i
dnn/zero_fraction_3/SizeSizednn/hiddenlayer_3/Relu*
T0*
out_type0	*
_output_shapes
: 
e
dnn/zero_fraction_3/LessEqual/yConst*
dtype0	*
_output_shapes
: *
valueB	 R????
?
dnn/zero_fraction_3/LessEqual	LessEqualdnn/zero_fraction_3/Sizednn/zero_fraction_3/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_3/cond/SwitchSwitchdnn/zero_fraction_3/LessEqualdnn/zero_fraction_3/LessEqual*
T0
*
_output_shapes
: : 
q
!dnn/zero_fraction_3/cond/switch_tIdentity!dnn/zero_fraction_3/cond/Switch:1*
_output_shapes
: *
T0

o
!dnn/zero_fraction_3/cond/switch_fIdentitydnn/zero_fraction_3/cond/Switch*
_output_shapes
: *
T0

l
 dnn/zero_fraction_3/cond/pred_idIdentitydnn/zero_fraction_3/LessEqual*
T0
*
_output_shapes
: 
?
,dnn/zero_fraction_3/cond/count_nonzero/zerosConst"^dnn/zero_fraction_3/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
/dnn/zero_fraction_3/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_3/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_3/cond/count_nonzero/zeros*
T0*'
_output_shapes
:?????????"
?
6dnn/zero_fraction_3/cond/count_nonzero/NotEqual/SwitchSwitchdnn/hiddenlayer_3/Relu dnn/zero_fraction_3/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_3/Relu*:
_output_shapes(
&:?????????":?????????"
?
+dnn/zero_fraction_3/cond/count_nonzero/CastCast/dnn/zero_fraction_3/cond/count_nonzero/NotEqual*

SrcT0
*'
_output_shapes
:?????????"*

DstT0
?
,dnn/zero_fraction_3/cond/count_nonzero/ConstConst"^dnn/zero_fraction_3/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
?
4dnn/zero_fraction_3/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_3/cond/count_nonzero/Cast,dnn/zero_fraction_3/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
dnn/zero_fraction_3/cond/CastCast4dnn/zero_fraction_3/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
?
.dnn/zero_fraction_3/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_3/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
?
1dnn/zero_fraction_3/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_3/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:?????????"
?
8dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/hiddenlayer_3/Relu dnn/zero_fraction_3/cond/pred_id*
T0*)
_class
loc:@dnn/hiddenlayer_3/Relu*:
_output_shapes(
&:?????????":?????????"
?
-dnn/zero_fraction_3/cond/count_nonzero_1/CastCast1dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual*

SrcT0
*'
_output_shapes
:?????????"*

DstT0	
?
.dnn/zero_fraction_3/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_3/cond/switch_f*
dtype0*
_output_shapes
:*
valueB"       
?
6dnn/zero_fraction_3/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_3/cond/count_nonzero_1/Cast.dnn/zero_fraction_3/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_3/cond/MergeMerge6dnn/zero_fraction_3/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_3/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
*dnn/zero_fraction_3/counts_to_fraction/subSubdnn/zero_fraction_3/Sizednn/zero_fraction_3/cond/Merge*
_output_shapes
: *
T0	
?
+dnn/zero_fraction_3/counts_to_fraction/CastCast*dnn/zero_fraction_3/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0

-dnn/zero_fraction_3/counts_to_fraction/Cast_1Castdnn/zero_fraction_3/Size*
_output_shapes
: *

DstT0*

SrcT0	
?
.dnn/zero_fraction_3/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_3/counts_to_fraction/Cast-dnn/zero_fraction_3/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_3/fractionIdentity.dnn/zero_fraction_3/counts_to_fraction/truediv*
_output_shapes
: *
T0
?
2dnn/dnn/hiddenlayer_3/fraction_of_zero_values/tagsConst*>
value5B3 B-dnn/dnn/hiddenlayer_3/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
-dnn/dnn/hiddenlayer_3/fraction_of_zero_valuesScalarSummary2dnn/dnn/hiddenlayer_3/fraction_of_zero_values/tagsdnn/zero_fraction_3/fraction*
T0*
_output_shapes
: 
?
$dnn/dnn/hiddenlayer_3/activation/tagConst*
dtype0*
_output_shapes
: *1
value(B& B dnn/dnn/hiddenlayer_3/activation
?
 dnn/dnn/hiddenlayer_3/activationHistogramSummary$dnn/dnn/hiddenlayer_3/activation/tagdnn/hiddenlayer_3/Relu*
_output_shapes
: 
?
9dnn/logits/kernel/part_0/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB""      *+
_class!
loc:@dnn/logits/kernel/part_0
?
7dnn/logits/kernel/part_0/Initializer/random_uniform/minConst*
valueB
 *??Ӿ*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
?
7dnn/logits/kernel/part_0/Initializer/random_uniform/maxConst*
valueB
 *???>*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
?
Adnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniformRandomUniform9dnn/logits/kernel/part_0/Initializer/random_uniform/shape*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes

:"
?
7dnn/logits/kernel/part_0/Initializer/random_uniform/subSub7dnn/logits/kernel/part_0/Initializer/random_uniform/max7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes
: 
?
7dnn/logits/kernel/part_0/Initializer/random_uniform/mulMulAdnn/logits/kernel/part_0/Initializer/random_uniform/RandomUniform7dnn/logits/kernel/part_0/Initializer/random_uniform/sub*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:"
?
3dnn/logits/kernel/part_0/Initializer/random_uniformAdd7dnn/logits/kernel/part_0/Initializer/random_uniform/mul7dnn/logits/kernel/part_0/Initializer/random_uniform/min*
T0*+
_class!
loc:@dnn/logits/kernel/part_0*
_output_shapes

:"
?
dnn/logits/kernel/part_0VarHandleOp*
shape
:"*)
shared_namednn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes
: 
?
9dnn/logits/kernel/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/kernel/part_0*
_output_shapes
: 
?
dnn/logits/kernel/part_0/AssignAssignVariableOpdnn/logits/kernel/part_03dnn/logits/kernel/part_0/Initializer/random_uniform*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0
?
,dnn/logits/kernel/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*+
_class!
loc:@dnn/logits/kernel/part_0*
dtype0*
_output_shapes

:"
?
(dnn/logits/bias/part_0/Initializer/zerosConst*
valueB*    *)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
?
dnn/logits/bias/part_0VarHandleOp*
shape:*'
shared_namednn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
: 
}
7dnn/logits/bias/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpdnn/logits/bias/part_0*
_output_shapes
: 
?
dnn/logits/bias/part_0/AssignAssignVariableOpdnn/logits/bias/part_0(dnn/logits/bias/part_0/Initializer/zeros*)
_class
loc:@dnn/logits/bias/part_0*
dtype0
?
*dnn/logits/bias/part_0/Read/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*)
_class
loc:@dnn/logits/bias/part_0*
dtype0*
_output_shapes
:
y
 dnn/logits/kernel/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
dtype0*
_output_shapes

:"
h
dnn/logits/kernelIdentity dnn/logits/kernel/ReadVariableOp*
T0*
_output_shapes

:"
x
dnn/logits/MatMulMatMuldnn/hiddenlayer_3/Reludnn/logits/kernel*
T0*'
_output_shapes
:?????????
q
dnn/logits/bias/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
`
dnn/logits/biasIdentitydnn/logits/bias/ReadVariableOp*
T0*
_output_shapes
:
s
dnn/logits/BiasAddBiasAdddnn/logits/MatMuldnn/logits/bias*
T0*'
_output_shapes
:?????????
e
dnn/zero_fraction_4/SizeSizednn/logits/BiasAdd*
T0*
out_type0	*
_output_shapes
: 
e
dnn/zero_fraction_4/LessEqual/yConst*
dtype0	*
_output_shapes
: *
valueB	 R????
?
dnn/zero_fraction_4/LessEqual	LessEqualdnn/zero_fraction_4/Sizednn/zero_fraction_4/LessEqual/y*
T0	*
_output_shapes
: 
?
dnn/zero_fraction_4/cond/SwitchSwitchdnn/zero_fraction_4/LessEqualdnn/zero_fraction_4/LessEqual*
_output_shapes
: : *
T0

q
!dnn/zero_fraction_4/cond/switch_tIdentity!dnn/zero_fraction_4/cond/Switch:1*
T0
*
_output_shapes
: 
o
!dnn/zero_fraction_4/cond/switch_fIdentitydnn/zero_fraction_4/cond/Switch*
T0
*
_output_shapes
: 
l
 dnn/zero_fraction_4/cond/pred_idIdentitydnn/zero_fraction_4/LessEqual*
T0
*
_output_shapes
: 
?
,dnn/zero_fraction_4/cond/count_nonzero/zerosConst"^dnn/zero_fraction_4/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
/dnn/zero_fraction_4/cond/count_nonzero/NotEqualNotEqual8dnn/zero_fraction_4/cond/count_nonzero/NotEqual/Switch:1,dnn/zero_fraction_4/cond/count_nonzero/zeros*
T0*'
_output_shapes
:?????????
?
6dnn/zero_fraction_4/cond/count_nonzero/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_4/cond/pred_id*
T0*%
_class
loc:@dnn/logits/BiasAdd*:
_output_shapes(
&:?????????:?????????
?
+dnn/zero_fraction_4/cond/count_nonzero/CastCast/dnn/zero_fraction_4/cond/count_nonzero/NotEqual*

SrcT0
*'
_output_shapes
:?????????*

DstT0
?
,dnn/zero_fraction_4/cond/count_nonzero/ConstConst"^dnn/zero_fraction_4/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
4dnn/zero_fraction_4/cond/count_nonzero/nonzero_countSum+dnn/zero_fraction_4/cond/count_nonzero/Cast,dnn/zero_fraction_4/cond/count_nonzero/Const*
_output_shapes
: *
T0
?
dnn/zero_fraction_4/cond/CastCast4dnn/zero_fraction_4/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
?
.dnn/zero_fraction_4/cond/count_nonzero_1/zerosConst"^dnn/zero_fraction_4/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
?
1dnn/zero_fraction_4/cond/count_nonzero_1/NotEqualNotEqual8dnn/zero_fraction_4/cond/count_nonzero_1/NotEqual/Switch.dnn/zero_fraction_4/cond/count_nonzero_1/zeros*
T0*'
_output_shapes
:?????????
?
8dnn/zero_fraction_4/cond/count_nonzero_1/NotEqual/SwitchSwitchdnn/logits/BiasAdd dnn/zero_fraction_4/cond/pred_id*
T0*%
_class
loc:@dnn/logits/BiasAdd*:
_output_shapes(
&:?????????:?????????
?
-dnn/zero_fraction_4/cond/count_nonzero_1/CastCast1dnn/zero_fraction_4/cond/count_nonzero_1/NotEqual*

SrcT0
*'
_output_shapes
:?????????*

DstT0	
?
.dnn/zero_fraction_4/cond/count_nonzero_1/ConstConst"^dnn/zero_fraction_4/cond/switch_f*
dtype0*
_output_shapes
:*
valueB"       
?
6dnn/zero_fraction_4/cond/count_nonzero_1/nonzero_countSum-dnn/zero_fraction_4/cond/count_nonzero_1/Cast.dnn/zero_fraction_4/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	
?
dnn/zero_fraction_4/cond/MergeMerge6dnn/zero_fraction_4/cond/count_nonzero_1/nonzero_countdnn/zero_fraction_4/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
*dnn/zero_fraction_4/counts_to_fraction/subSubdnn/zero_fraction_4/Sizednn/zero_fraction_4/cond/Merge*
_output_shapes
: *
T0	
?
+dnn/zero_fraction_4/counts_to_fraction/CastCast*dnn/zero_fraction_4/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	

-dnn/zero_fraction_4/counts_to_fraction/Cast_1Castdnn/zero_fraction_4/Size*
_output_shapes
: *

DstT0*

SrcT0	
?
.dnn/zero_fraction_4/counts_to_fraction/truedivRealDiv+dnn/zero_fraction_4/counts_to_fraction/Cast-dnn/zero_fraction_4/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
y
dnn/zero_fraction_4/fractionIdentity.dnn/zero_fraction_4/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
+dnn/dnn/logits/fraction_of_zero_values/tagsConst*7
value.B, B&dnn/dnn/logits/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
&dnn/dnn/logits/fraction_of_zero_valuesScalarSummary+dnn/dnn/logits/fraction_of_zero_values/tagsdnn/zero_fraction_4/fraction*
T0*
_output_shapes
: 
w
dnn/dnn/logits/activation/tagConst**
value!B Bdnn/dnn/logits/activation*
dtype0*
_output_shapes
: 
x
dnn/dnn/logits/activationHistogramSummarydnn/dnn/logits/activation/tagdnn/logits/BiasAdd*
_output_shapes
: 
?
Clinear/linear_model/age_bucketized/weights/part_0/Initializer/zerosConst*
valueB*    *D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*
dtype0*
_output_shapes

:
?
1linear/linear_model/age_bucketized/weights/part_0VarHandleOp*
shape
:*B
shared_name31linear/linear_model/age_bucketized/weights/part_0*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*
dtype0*
_output_shapes
: 
?
Rlinear/linear_model/age_bucketized/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp1linear/linear_model/age_bucketized/weights/part_0*
_output_shapes
: 
?
8linear/linear_model/age_bucketized/weights/part_0/AssignAssignVariableOp1linear/linear_model/age_bucketized/weights/part_0Clinear/linear_model/age_bucketized/weights/part_0/Initializer/zeros*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*
dtype0
?
Elinear/linear_model/age_bucketized/weights/part_0/Read/ReadVariableOpReadVariableOp1linear/linear_model/age_bucketized/weights/part_0*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*
dtype0*
_output_shapes

:
?
glinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"@B    *X
_classN
LJloc:@linear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0
?
]linear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/Initializer/zeros/ConstConst*
dtype0*
_output_shapes
: *
valueB
 *    *X
_classN
LJloc:@linear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0
?
Wlinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/Initializer/zerosFillglinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/Initializer/zeros/shape_as_tensor]linear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/Initializer/zeros/Const*
T0*X
_classN
LJloc:@linear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0* 
_output_shapes
:
??=
?
Elinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape:
??=*V
shared_nameGElinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0*X
_classN
LJloc:@linear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0
?
flinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOpElinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0*
_output_shapes
: 
?
Llinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/AssignAssignVariableOpElinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0Wlinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/Initializer/zeros*
dtype0*X
_classN
LJloc:@linear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0
?
Ylinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/Read/ReadVariableOpReadVariableOpElinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0*X
_classN
LJloc:@linear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0*
dtype0* 
_output_shapes
:
??=
?
>linear/linear_model/education/weights/part_0/Initializer/zerosConst*
valueB*    *?
_class5
31loc:@linear/linear_model/education/weights/part_0*
dtype0*
_output_shapes

:
?
,linear/linear_model/education/weights/part_0VarHandleOp*?
_class5
31loc:@linear/linear_model/education/weights/part_0*
dtype0*
_output_shapes
: *
shape
:*=
shared_name.,linear/linear_model/education/weights/part_0
?
Mlinear/linear_model/education/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp,linear/linear_model/education/weights/part_0*
_output_shapes
: 
?
3linear/linear_model/education/weights/part_0/AssignAssignVariableOp,linear/linear_model/education/weights/part_0>linear/linear_model/education/weights/part_0/Initializer/zeros*?
_class5
31loc:@linear/linear_model/education/weights/part_0*
dtype0
?
@linear/linear_model/education/weights/part_0/Read/ReadVariableOpReadVariableOp,linear/linear_model/education/weights/part_0*?
_class5
31loc:@linear/linear_model/education/weights/part_0*
dtype0*
_output_shapes

:
?
[linear/linear_model/education_X_occupation/weights/part_0/Initializer/zeros/shape_as_tensorConst*
valueB"'     *L
_classB
@>loc:@linear/linear_model/education_X_occupation/weights/part_0*
dtype0*
_output_shapes
:
?
Qlinear/linear_model/education_X_occupation/weights/part_0/Initializer/zeros/ConstConst*
valueB
 *    *L
_classB
@>loc:@linear/linear_model/education_X_occupation/weights/part_0*
dtype0*
_output_shapes
: 
?
Klinear/linear_model/education_X_occupation/weights/part_0/Initializer/zerosFill[linear/linear_model/education_X_occupation/weights/part_0/Initializer/zeros/shape_as_tensorQlinear/linear_model/education_X_occupation/weights/part_0/Initializer/zeros/Const*
T0*L
_classB
@>loc:@linear/linear_model/education_X_occupation/weights/part_0*
_output_shapes
:	?N
?
9linear/linear_model/education_X_occupation/weights/part_0VarHandleOp*
shape:	?N*J
shared_name;9linear/linear_model/education_X_occupation/weights/part_0*L
_classB
@>loc:@linear/linear_model/education_X_occupation/weights/part_0*
dtype0*
_output_shapes
: 
?
Zlinear/linear_model/education_X_occupation/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp9linear/linear_model/education_X_occupation/weights/part_0*
_output_shapes
: 
?
@linear/linear_model/education_X_occupation/weights/part_0/AssignAssignVariableOp9linear/linear_model/education_X_occupation/weights/part_0Klinear/linear_model/education_X_occupation/weights/part_0/Initializer/zeros*L
_classB
@>loc:@linear/linear_model/education_X_occupation/weights/part_0*
dtype0
?
Mlinear/linear_model/education_X_occupation/weights/part_0/Read/ReadVariableOpReadVariableOp9linear/linear_model/education_X_occupation/weights/part_0*
dtype0*
_output_shapes
:	?N*L
_classB
@>loc:@linear/linear_model/education_X_occupation/weights/part_0
?
;linear/linear_model/gender/weights/part_0/Initializer/zerosConst*
dtype0*
_output_shapes

:*
valueB*    *<
_class2
0.loc:@linear/linear_model/gender/weights/part_0
?
)linear/linear_model/gender/weights/part_0VarHandleOp*
dtype0*
_output_shapes
: *
shape
:*:
shared_name+)linear/linear_model/gender/weights/part_0*<
_class2
0.loc:@linear/linear_model/gender/weights/part_0
?
Jlinear/linear_model/gender/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp)linear/linear_model/gender/weights/part_0*
_output_shapes
: 
?
0linear/linear_model/gender/weights/part_0/AssignAssignVariableOp)linear/linear_model/gender/weights/part_0;linear/linear_model/gender/weights/part_0/Initializer/zeros*<
_class2
0.loc:@linear/linear_model/gender/weights/part_0*
dtype0
?
=linear/linear_model/gender/weights/part_0/Read/ReadVariableOpReadVariableOp)linear/linear_model/gender/weights/part_0*<
_class2
0.loc:@linear/linear_model/gender/weights/part_0*
dtype0*
_output_shapes

:
?
Clinear/linear_model/marital_status/weights/part_0/Initializer/zerosConst*
valueB*    *D
_class:
86loc:@linear/linear_model/marital_status/weights/part_0*
dtype0*
_output_shapes

:
?
1linear/linear_model/marital_status/weights/part_0VarHandleOp*B
shared_name31linear/linear_model/marital_status/weights/part_0*D
_class:
86loc:@linear/linear_model/marital_status/weights/part_0*
dtype0*
_output_shapes
: *
shape
:
?
Rlinear/linear_model/marital_status/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp1linear/linear_model/marital_status/weights/part_0*
_output_shapes
: 
?
8linear/linear_model/marital_status/weights/part_0/AssignAssignVariableOp1linear/linear_model/marital_status/weights/part_0Clinear/linear_model/marital_status/weights/part_0/Initializer/zeros*D
_class:
86loc:@linear/linear_model/marital_status/weights/part_0*
dtype0
?
Elinear/linear_model/marital_status/weights/part_0/Read/ReadVariableOpReadVariableOp1linear/linear_model/marital_status/weights/part_0*
dtype0*
_output_shapes

:*D
_class:
86loc:@linear/linear_model/marital_status/weights/part_0
?
Clinear/linear_model/native_country/weights/part_0/Initializer/zerosConst*
valueBd*    *D
_class:
86loc:@linear/linear_model/native_country/weights/part_0*
dtype0*
_output_shapes

:d
?
1linear/linear_model/native_country/weights/part_0VarHandleOp*
shape
:d*B
shared_name31linear/linear_model/native_country/weights/part_0*D
_class:
86loc:@linear/linear_model/native_country/weights/part_0*
dtype0*
_output_shapes
: 
?
Rlinear/linear_model/native_country/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp1linear/linear_model/native_country/weights/part_0*
_output_shapes
: 
?
8linear/linear_model/native_country/weights/part_0/AssignAssignVariableOp1linear/linear_model/native_country/weights/part_0Clinear/linear_model/native_country/weights/part_0/Initializer/zeros*D
_class:
86loc:@linear/linear_model/native_country/weights/part_0*
dtype0
?
Elinear/linear_model/native_country/weights/part_0/Read/ReadVariableOpReadVariableOp1linear/linear_model/native_country/weights/part_0*
dtype0*
_output_shapes

:d*D
_class:
86loc:@linear/linear_model/native_country/weights/part_0
?
`linear/linear_model/native_country_X_occupation/weights/part_0/Initializer/zeros/shape_as_tensorConst*
dtype0*
_output_shapes
:*
valueB"'     *Q
_classG
ECloc:@linear/linear_model/native_country_X_occupation/weights/part_0
?
Vlinear/linear_model/native_country_X_occupation/weights/part_0/Initializer/zeros/ConstConst*
valueB
 *    *Q
_classG
ECloc:@linear/linear_model/native_country_X_occupation/weights/part_0*
dtype0*
_output_shapes
: 
?
Plinear/linear_model/native_country_X_occupation/weights/part_0/Initializer/zerosFill`linear/linear_model/native_country_X_occupation/weights/part_0/Initializer/zeros/shape_as_tensorVlinear/linear_model/native_country_X_occupation/weights/part_0/Initializer/zeros/Const*
T0*Q
_classG
ECloc:@linear/linear_model/native_country_X_occupation/weights/part_0*
_output_shapes
:	?N
?
>linear/linear_model/native_country_X_occupation/weights/part_0VarHandleOp*Q
_classG
ECloc:@linear/linear_model/native_country_X_occupation/weights/part_0*
dtype0*
_output_shapes
: *
shape:	?N*O
shared_name@>linear/linear_model/native_country_X_occupation/weights/part_0
?
_linear/linear_model/native_country_X_occupation/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp>linear/linear_model/native_country_X_occupation/weights/part_0*
_output_shapes
: 
?
Elinear/linear_model/native_country_X_occupation/weights/part_0/AssignAssignVariableOp>linear/linear_model/native_country_X_occupation/weights/part_0Plinear/linear_model/native_country_X_occupation/weights/part_0/Initializer/zeros*Q
_classG
ECloc:@linear/linear_model/native_country_X_occupation/weights/part_0*
dtype0
?
Rlinear/linear_model/native_country_X_occupation/weights/part_0/Read/ReadVariableOpReadVariableOp>linear/linear_model/native_country_X_occupation/weights/part_0*
dtype0*
_output_shapes
:	?N*Q
_classG
ECloc:@linear/linear_model/native_country_X_occupation/weights/part_0
?
?linear/linear_model/occupation/weights/part_0/Initializer/zerosConst*
valueBd*    *@
_class6
42loc:@linear/linear_model/occupation/weights/part_0*
dtype0*
_output_shapes

:d
?
-linear/linear_model/occupation/weights/part_0VarHandleOp*@
_class6
42loc:@linear/linear_model/occupation/weights/part_0*
dtype0*
_output_shapes
: *
shape
:d*>
shared_name/-linear/linear_model/occupation/weights/part_0
?
Nlinear/linear_model/occupation/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp-linear/linear_model/occupation/weights/part_0*
_output_shapes
: 
?
4linear/linear_model/occupation/weights/part_0/AssignAssignVariableOp-linear/linear_model/occupation/weights/part_0?linear/linear_model/occupation/weights/part_0/Initializer/zeros*
dtype0*@
_class6
42loc:@linear/linear_model/occupation/weights/part_0
?
Alinear/linear_model/occupation/weights/part_0/Read/ReadVariableOpReadVariableOp-linear/linear_model/occupation/weights/part_0*
dtype0*
_output_shapes

:d*@
_class6
42loc:@linear/linear_model/occupation/weights/part_0
?
Alinear/linear_model/relationship/weights/part_0/Initializer/zerosConst*
valueB*    *B
_class8
64loc:@linear/linear_model/relationship/weights/part_0*
dtype0*
_output_shapes

:
?
/linear/linear_model/relationship/weights/part_0VarHandleOp*B
_class8
64loc:@linear/linear_model/relationship/weights/part_0*
dtype0*
_output_shapes
: *
shape
:*@
shared_name1/linear/linear_model/relationship/weights/part_0
?
Plinear/linear_model/relationship/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp/linear/linear_model/relationship/weights/part_0*
_output_shapes
: 
?
6linear/linear_model/relationship/weights/part_0/AssignAssignVariableOp/linear/linear_model/relationship/weights/part_0Alinear/linear_model/relationship/weights/part_0/Initializer/zeros*
dtype0*B
_class8
64loc:@linear/linear_model/relationship/weights/part_0
?
Clinear/linear_model/relationship/weights/part_0/Read/ReadVariableOpReadVariableOp/linear/linear_model/relationship/weights/part_0*B
_class8
64loc:@linear/linear_model/relationship/weights/part_0*
dtype0*
_output_shapes

:
?
>linear/linear_model/workclass/weights/part_0/Initializer/zerosConst*
valueB	*    *?
_class5
31loc:@linear/linear_model/workclass/weights/part_0*
dtype0*
_output_shapes

:	
?
,linear/linear_model/workclass/weights/part_0VarHandleOp*
shape
:	*=
shared_name.,linear/linear_model/workclass/weights/part_0*?
_class5
31loc:@linear/linear_model/workclass/weights/part_0*
dtype0*
_output_shapes
: 
?
Mlinear/linear_model/workclass/weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp,linear/linear_model/workclass/weights/part_0*
_output_shapes
: 
?
3linear/linear_model/workclass/weights/part_0/AssignAssignVariableOp,linear/linear_model/workclass/weights/part_0>linear/linear_model/workclass/weights/part_0/Initializer/zeros*?
_class5
31loc:@linear/linear_model/workclass/weights/part_0*
dtype0
?
@linear/linear_model/workclass/weights/part_0/Read/ReadVariableOpReadVariableOp,linear/linear_model/workclass/weights/part_0*?
_class5
31loc:@linear/linear_model/workclass/weights/part_0*
dtype0*
_output_shapes

:	
?
9linear/linear_model/bias_weights/part_0/Initializer/zerosConst*
dtype0*
_output_shapes
:*
valueB*    *:
_class0
.,loc:@linear/linear_model/bias_weights/part_0
?
'linear/linear_model/bias_weights/part_0VarHandleOp*8
shared_name)'linear/linear_model/bias_weights/part_0*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
: *
shape:
?
Hlinear/linear_model/bias_weights/part_0/IsInitialized/VarIsInitializedOpVarIsInitializedOp'linear/linear_model/bias_weights/part_0*
_output_shapes
: 
?
.linear/linear_model/bias_weights/part_0/AssignAssignVariableOp'linear/linear_model/bias_weights/part_09linear/linear_model/bias_weights/part_0/Initializer/zeros*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0*
dtype0
?
;linear/linear_model/bias_weights/part_0/Read/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:*:
_class0
.,loc:@linear/linear_model/bias_weights/part_0
?
Klinear/linear_model/linear_model/linear_model/age_bucketized/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Glinear/linear_model/linear_model/linear_model/age_bucketized/ExpandDims
ExpandDimsPlaceholder_8Klinear/linear_model/linear_model/linear_model/age_bucketized/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
Flinear/linear_model/linear_model/linear_model/age_bucketized/Bucketize	BucketizeGlinear/linear_model/linear_model/linear_model/age_bucketized/ExpandDims*:

boundaries,
*"(  ?A  ?A  ?A  B   B  4B  HB  \B  pB  ?B*
T0*'
_output_shapes
:?????????
?
Blinear/linear_model/linear_model/linear_model/age_bucketized/ShapeShapeFlinear/linear_model/linear_model/linear_model/age_bucketized/Bucketize*
_output_shapes
:*
T0
?
Plinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
?
Rlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/age_bucketized/strided_sliceStridedSliceBlinear/linear_model/linear_model/linear_model/age_bucketized/ShapePlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice/stackRlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice/stack_1Rlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
?
Hlinear/linear_model/linear_model/linear_model/age_bucketized/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
?
Hlinear/linear_model/linear_model/linear_model/age_bucketized/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
Blinear/linear_model/linear_model/linear_model/age_bucketized/rangeRangeHlinear/linear_model/linear_model/linear_model/age_bucketized/range/startJlinear/linear_model/linear_model/linear_model/age_bucketized/strided_sliceHlinear/linear_model/linear_model/linear_model/age_bucketized/range/delta*#
_output_shapes
:?????????
?
Mlinear/linear_model/linear_model/linear_model/age_bucketized/ExpandDims_1/dimConst*
dtype0*
_output_shapes
: *
value	B :
?
Ilinear/linear_model/linear_model/linear_model/age_bucketized/ExpandDims_1
ExpandDimsBlinear/linear_model/linear_model/linear_model/age_bucketized/rangeMlinear/linear_model/linear_model/linear_model/age_bucketized/ExpandDims_1/dim*'
_output_shapes
:?????????*
T0
?
Klinear/linear_model/linear_model/linear_model/age_bucketized/Tile/multiplesConst*
valueB"      *
dtype0*
_output_shapes
:
?
Alinear/linear_model/linear_model/linear_model/age_bucketized/TileTileIlinear/linear_model/linear_model/linear_model/age_bucketized/ExpandDims_1Klinear/linear_model/linear_model/linear_model/age_bucketized/Tile/multiples*'
_output_shapes
:?????????*
T0
?
Jlinear/linear_model/linear_model/linear_model/age_bucketized/Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Dlinear/linear_model/linear_model/linear_model/age_bucketized/ReshapeReshapeAlinear/linear_model/linear_model/linear_model/age_bucketized/TileJlinear/linear_model/linear_model/linear_model/age_bucketized/Reshape/shape*
T0*#
_output_shapes
:?????????
?
Jlinear/linear_model/linear_model/linear_model/age_bucketized/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
?
Jlinear/linear_model/linear_model/linear_model/age_bucketized/range_1/limitConst*
value	B :*
dtype0*
_output_shapes
: 
?
Jlinear/linear_model/linear_model/linear_model/age_bucketized/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
Dlinear/linear_model/linear_model/linear_model/age_bucketized/range_1RangeJlinear/linear_model/linear_model/linear_model/age_bucketized/range_1/startJlinear/linear_model/linear_model/linear_model/age_bucketized/range_1/limitJlinear/linear_model/linear_model/linear_model/age_bucketized/range_1/delta*
_output_shapes
:
?
Mlinear/linear_model/linear_model/linear_model/age_bucketized/Tile_1/multiplesPackJlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice*
T0*
N*
_output_shapes
:
?
Clinear/linear_model/linear_model/linear_model/age_bucketized/Tile_1TileDlinear/linear_model/linear_model/linear_model/age_bucketized/range_1Mlinear/linear_model/linear_model/linear_model/age_bucketized/Tile_1/multiples*
T0*#
_output_shapes
:?????????
?
Llinear/linear_model/linear_model/linear_model/age_bucketized/Reshape_1/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Flinear/linear_model/linear_model/linear_model/age_bucketized/Reshape_1ReshapeFlinear/linear_model/linear_model/linear_model/age_bucketized/BucketizeLlinear/linear_model/linear_model/linear_model/age_bucketized/Reshape_1/shape*#
_output_shapes
:?????????*
T0
?
Blinear/linear_model/linear_model/linear_model/age_bucketized/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
?
@linear/linear_model/linear_model/linear_model/age_bucketized/mulMulBlinear/linear_model/linear_model/linear_model/age_bucketized/mul/xClinear/linear_model/linear_model/linear_model/age_bucketized/Tile_1*
T0*#
_output_shapes
:?????????
?
@linear/linear_model/linear_model/linear_model/age_bucketized/addAddFlinear/linear_model/linear_model/linear_model/age_bucketized/Reshape_1@linear/linear_model/linear_model/linear_model/age_bucketized/mul*
T0*#
_output_shapes
:?????????
?
Blinear/linear_model/linear_model/linear_model/age_bucketized/stackPackDlinear/linear_model/linear_model/linear_model/age_bucketized/ReshapeClinear/linear_model/linear_model/linear_model/age_bucketized/Tile_1*
T0*
N*'
_output_shapes
:?????????
?
Klinear/linear_model/linear_model/linear_model/age_bucketized/transpose/permConst*
valueB"       *
dtype0*
_output_shapes
:
?
Flinear/linear_model/linear_model/linear_model/age_bucketized/transpose	TransposeBlinear/linear_model/linear_model/linear_model/age_bucketized/stackKlinear/linear_model/linear_model/linear_model/age_bucketized/transpose/perm*'
_output_shapes
:?????????*
T0
?
Alinear/linear_model/linear_model/linear_model/age_bucketized/CastCastFlinear/linear_model/linear_model/linear_model/age_bucketized/transpose*'
_output_shapes
:?????????*

DstT0	*

SrcT0
?
Flinear/linear_model/linear_model/linear_model/age_bucketized/stack_1/1Const*
dtype0*
_output_shapes
: *
value	B :
?
Dlinear/linear_model/linear_model/linear_model/age_bucketized/stack_1PackJlinear/linear_model/linear_model/linear_model/age_bucketized/strided_sliceFlinear/linear_model/linear_model/linear_model/age_bucketized/stack_1/1*
T0*
N*
_output_shapes
:
?
Clinear/linear_model/linear_model/linear_model/age_bucketized/Cast_1CastDlinear/linear_model/linear_model/linear_model/age_bucketized/stack_1*

SrcT0*
_output_shapes
:*

DstT0	
?
Ilinear/linear_model/linear_model/linear_model/age_bucketized/Shape_1/CastCastClinear/linear_model/linear_model/linear_model/age_bucketized/Cast_1*

SrcT0	*
_output_shapes
:*

DstT0
?
Rlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: 
?
Tlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Tlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Llinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1StridedSliceIlinear/linear_model/linear_model/linear_model/age_bucketized/Shape_1/CastRlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1/stackTlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1/stack_1Tlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
?
Glinear/linear_model/linear_model/linear_model/age_bucketized/Cast_2/x/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Elinear/linear_model/linear_model/linear_model/age_bucketized/Cast_2/xPackLlinear/linear_model/linear_model/linear_model/age_bucketized/strided_slice_1Glinear/linear_model/linear_model/linear_model/age_bucketized/Cast_2/x/1*
N*
_output_shapes
:*
T0
?
Clinear/linear_model/linear_model/linear_model/age_bucketized/Cast_2CastElinear/linear_model/linear_model/linear_model/age_bucketized/Cast_2/x*

SrcT0*
_output_shapes
:*

DstT0	
?
Jlinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshapeSparseReshapeAlinear/linear_model/linear_model/linear_model/age_bucketized/CastClinear/linear_model/linear_model/linear_model/age_bucketized/Cast_1Clinear/linear_model/linear_model/linear_model/age_bucketized/Cast_2*-
_output_shapes
:?????????:
?
Slinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshape/IdentityIdentity@linear/linear_model/linear_model/linear_model/age_bucketized/add*
T0*#
_output_shapes
:?????????
?
Ulinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
Tlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SliceSliceLlinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshape:1Ulinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice/beginTlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
?
Nlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/ProdProdOlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SliceOlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Const*
_output_shapes
: *
T0	
?
Zlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Rlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2GatherV2Llinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshape:1Zlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2/indicesWlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Taxis0*
Tindices0
?
Plinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Cast/xPackNlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/ProdRlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2*
T0	*
N*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseReshapeSparseReshapeJlinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshapeLlinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshape:1Plinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
`linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseReshape/IdentityIdentitySlinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshape/Identity*
T0*#
_output_shapes
:?????????
?
Xlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GreaterEqual/yConst*
value	B : *
dtype0*
_output_shapes
: 
?
Vlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GreaterEqualGreaterEqual`linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseReshape/IdentityXlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GreaterEqual/y*
T0*#
_output_shapes
:?????????
?
Olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/WhereWhereVlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/ReshapeReshapeOlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/WhereWlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape/shape*#
_output_shapes
:?????????*
T0	
?
Ylinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Tlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_1GatherV2Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseReshapeQlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/ReshapeYlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_1/axis*'
_output_shapes
:?????????*
Taxis0*
Tindices0	*
Tparams0	
?
Ylinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Tlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_2GatherV2`linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseReshape/IdentityQlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/ReshapeYlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_2/axis*
Tindices0	*
Tparams0*#
_output_shapes
:?????????*
Taxis0
?
Rlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/IdentityIdentityYlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
clinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/ConstConst*
value	B : *
dtype0*
_output_shapes
: 
?
qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsTlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_1Tlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/GatherV2_2Rlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Identityclinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/Const*
T0*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
ulinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
?
wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
?
olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceqlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsulinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stackwlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*#
_output_shapes
:?????????*
Index0*
T0	*
shrink_axis_mask*

begin_mask
?
flinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/CastCastolinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/strided_slice*#
_output_shapes
:?????????*

DstT0*

SrcT0	
?
hlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/UniqueUniqueslinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0*2
_output_shapes 
:?????????:?????????
?
rlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather1linear/linear_model/age_bucketized/weights/part_0hlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*
dtype0*'
_output_shapes
:?????????
?
{linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentityrlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*'
_output_shapes
:?????????
?
}linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identity{linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
alinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparseSparseSegmentSum}linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1jlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/Unique:1flinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:?????????
?
Ylinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape_1/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape_1Reshapeslinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Ylinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape_1/shape*'
_output_shapes
:?????????*
T0

?
Olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/ShapeShapealinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
]linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
_linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
_linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_sliceStridedSliceOlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Shape]linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_slice/stack_linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_slice/stack_1_linear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
?
Olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/stackPackQlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/stack/0Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/strided_slice*
T0*
N*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/TileTileSlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape_1Olinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
Tlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/zeros_like	ZerosLikealinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Ilinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sumSelectNlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/TileTlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/zeros_likealinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Plinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Cast_1CastLlinear/linear_model/linear_model/linear_model/age_bucketized/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
Vlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_1SlicePlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Cast_1Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_1/beginVlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Shape_1ShapeIlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum*
T0*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
?
Vlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_2/sizeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_2SliceQlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Shape_1Wlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_2/beginVlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
Ulinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Plinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/concatConcatV2Qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_1Qlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Slice_2Ulinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/concat/axis*
T0*
N*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape_2ReshapeIlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sumPlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/concat*'
_output_shapes
:?????????*
T0
?
Vlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ShapeShapeFlinear/linear_model/linear_model/linear_model/age_bucketized/Bucketize*
T0*
_output_shapes
:
?
dlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
flinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
flinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_sliceStridedSliceVlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Shapedlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice/stackflinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice/stack_1flinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
T0*
Index0
?
\linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
?
\linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
Vlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/rangeRange\linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/range/start^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice\linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/range/delta*#
_output_shapes
:?????????
?
_linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims/dimConst*
value	B :*
dtype0*
_output_shapes
: 
?
[linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims
ExpandDimsVlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/range_linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
_linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Tile/multiplesConst*
valueB"      *
dtype0*
_output_shapes
:
?
Ulinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/TileTile[linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Tile/multiples*
T0*'
_output_shapes
:?????????
?
^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Xlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ReshapeReshapeUlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Tile^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Reshape/shape*#
_output_shapes
:?????????*
T0
?
^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
?
^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/range_1/limitConst*
value	B :*
dtype0*
_output_shapes
: 
?
^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
Xlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/range_1Range^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/range_1/start^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/range_1/limit^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/range_1/delta*
_output_shapes
:
?
alinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Tile_1/multiplesPack^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice*
T0*
N*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Tile_1TileXlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/range_1alinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Tile_1/multiples*
T0*#
_output_shapes
:?????????
?
`linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Reshape_1/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Zlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Reshape_1ReshapeFlinear/linear_model/linear_model/linear_model/age_bucketized/Bucketize`linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Reshape_1/shape*
T0*#
_output_shapes
:?????????
?
Vlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/mul/xConst*
value	B :*
dtype0*
_output_shapes
: 
?
Tlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/mulMulVlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/mul/xWlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Tile_1*#
_output_shapes
:?????????*
T0
?
Tlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/addAddZlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Reshape_1Tlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/mul*#
_output_shapes
:?????????*
T0
?
Vlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/stackPackXlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ReshapeWlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Tile_1*
T0*
N*'
_output_shapes
:?????????
?
_linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/transpose/permConst*
dtype0*
_output_shapes
:*
valueB"       
?
Zlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/transpose	TransposeVlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/stack_linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/transpose/perm*
T0*'
_output_shapes
:?????????
?
Ulinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/CastCastZlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/transpose*'
_output_shapes
:?????????*

DstT0	*

SrcT0
?
Zlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/stack_1/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
Xlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/stack_1Pack^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_sliceZlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/stack_1/1*
T0*
N*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Cast_1CastXlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/stack_1*

SrcT0*
_output_shapes
:*

DstT0	
?
alinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_1/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_1
ExpandDimsPlaceholder_1alinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_1/dim*
T0*'
_output_shapes
:?????????
?
olinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
ilinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/to_sparse_input/NotEqualNotEqual]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_1olinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/to_sparse_input/ignore_value/x*'
_output_shapes
:?????????*
T0
?
hlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/to_sparse_input/indicesWhereilinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
glinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/to_sparse_input/valuesGatherNd]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_1hlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:?????????
?
llinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/to_sparse_input/dense_shapeShape]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_1*
T0*
out_type0	*
_output_shapes
:
?
blinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/ConstConst*U
valueLBJB Amer-Indian-EskimoB Asian-Pac-IslanderB BlackB OtherB White*
dtype0*
_output_shapes
:
?
alinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
?
hlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
?
hlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
?
blinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/rangeRangehlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/range/startalinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/Sizehlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/range/delta*
_output_shapes
:
?
alinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/CastCastblinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/range*
_output_shapes
:*

DstT0	*

SrcT0
?
mlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/hash_table/ConstConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
rlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/hash_table/hash_tableHashTableV2*
value_dtype0	*
	key_dtype0*
_output_shapes
: 
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2rlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/hash_table/hash_tableblinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/Constalinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/Cast*	
Tin0*

Tout0	
?
tlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/hash_table_Lookup/LookupTableFindV2LookupTableFindV2rlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/hash_table/hash_tableglinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/to_sparse_input/valuesmlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/hash_table/Const*#
_output_shapes
:?????????*	
Tin0*

Tout0	
?
alinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_2/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_2
ExpandDimsPlaceholder_6alinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_2/dim*
T0*'
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Cast_2CastTlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/add*

SrcT0*#
_output_shapes
:?????????*

DstT0	
?
\linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/SparseCrossSparseCrossUlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Casthlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/to_sparse_input/indicesWlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Cast_2tlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/hash_table_Lookup/LookupTableFindV2Wlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Cast_1llinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/to_sparse_input/dense_shape]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_2*
num_buckets??=*
hashed_output(*
out_type0	*
N*<
_output_shapes*
(:?????????:?????????:*
dense_types
2*
hash_key?????*
internal_type0	*
sparse_types
2		
?
]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Shape_1/CastCast^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/SparseCross:2*

SrcT0	*
_output_shapes
:*

DstT0
?
flinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice_1/stackConst*
dtype0*
_output_shapes
:*
valueB: 
?
hlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
hlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
`linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice_1StridedSlice]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Shape_1/Castflinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice_1/stackhlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice_1/stack_1hlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
?
[linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Cast_3/x/1Const*
dtype0*
_output_shapes
: *
valueB :
?????????
?
Ylinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Cast_3/xPack`linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/strided_slice_1[linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Cast_3/x/1*
T0*
N*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Cast_3CastYlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Cast_3/x*
_output_shapes
:*

DstT0	*

SrcT0
?
^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/SparseReshapeSparseReshape\linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/SparseCross^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/SparseCross:2Wlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/Cast_3*-
_output_shapes
:?????????:
?
glinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/SparseReshape/IdentityIdentity^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/SparseCross:1*#
_output_shapes
:?????????*
T0	
?
ilinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
?
hlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
clinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/SliceSlice`linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/SparseReshape:1ilinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice/beginhlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice/size*
_output_shapes
:*
Index0*
T0	
?
clinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
blinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/ProdProdclinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Sliceclinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Const*
T0	*
_output_shapes
: 
?
nlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
klinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
flinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GatherV2GatherV2`linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/SparseReshape:1nlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GatherV2/indicesklinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Taxis0*
Tindices0
?
dlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Cast/xPackblinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Prodflinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GatherV2*
T0	*
N*
_output_shapes
:
?
klinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/SparseReshapeSparseReshape^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/SparseReshape`linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/SparseReshape:1dlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
tlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/SparseReshape/IdentityIdentityglinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
llinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
?
jlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GreaterEqualGreaterEqualtlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/SparseReshape/Identityllinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GreaterEqual/y*#
_output_shapes
:?????????*
T0	
?
clinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/WhereWherejlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
klinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
elinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/ReshapeReshapeclinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Whereklinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
mlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
hlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GatherV2_1GatherV2klinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/SparseReshapeelinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Reshapemlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GatherV2_1/axis*
Taxis0*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????
?
mlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
hlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GatherV2_2GatherV2tlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/SparseReshape/Identityelinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Reshapemlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GatherV2_2/axis*#
_output_shapes
:?????????*
Taxis0*
Tindices0	*
Tparams0	
?
flinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/IdentityIdentitymlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
wlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/SparseFillEmptyRows/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowshlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GatherV2_1hlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/GatherV2_2flinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Identitywlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB"       
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSlice?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/strided_slice/stack?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:?????????
?
zlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/CastCast?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:?????????*

DstT0
?
|linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/UniqueUnique?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGatherElinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0|linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/Unique*X
_classN
LJloc:@linear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0*
dtype0*'
_output_shapes
:?????????*
Tindices0	
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentity?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:?????????*
T0*X
_classN
LJloc:@linear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0
?
?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identity?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*'
_output_shapes
:?????????*
T0
?
ulinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparseSparseSegmentSum?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1~linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/Unique:1zlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse/Cast*'
_output_shapes
:?????????*
T0
?
mlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Reshape_1/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
?
glinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Reshape_1Reshape?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2mlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Reshape_1/shape*'
_output_shapes
:?????????*
T0

?
clinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/ShapeShapeulinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
qlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
slinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
slinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
klinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/strided_sliceStridedSliceclinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Shapeqlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/strided_slice/stackslinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/strided_slice/stack_1slinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
?
elinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
?
clinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/stackPackelinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/stack/0klinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/strided_slice*
T0*
N*
_output_shapes
:
?
blinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/TileTileglinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Reshape_1clinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
hlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/zeros_like	ZerosLikeulinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sumSelectblinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Tilehlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/zeros_likeulinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
dlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Cast_1Cast`linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
?
klinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
jlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
?
elinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice_1Slicedlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Cast_1klinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice_1/beginjlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
elinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Shape_1Shape]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum*
_output_shapes
:*
T0
?
klinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
?
jlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice_2/sizeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
elinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice_2Sliceelinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Shape_1klinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice_2/beginjlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
ilinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
dlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/concatConcatV2elinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice_1elinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Slice_2ilinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/concat/axis*
N*
_output_shapes
:*
T0
?
glinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Reshape_2Reshape]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sumdlinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/concat*'
_output_shapes
:?????????*
T0
?
Flinear/linear_model/linear_model/linear_model/education/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Blinear/linear_model/linear_model/linear_model/education/ExpandDims
ExpandDimsPlaceholder_2Flinear/linear_model/linear_model/linear_model/education/ExpandDims/dim*'
_output_shapes
:?????????*
T0
?
Vlinear/linear_model/linear_model/linear_model/education/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
Plinear/linear_model/linear_model/linear_model/education/to_sparse_input/NotEqualNotEqualBlinear/linear_model/linear_model/linear_model/education/ExpandDimsVlinear/linear_model/linear_model/linear_model/education/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:?????????
?
Olinear/linear_model/linear_model/linear_model/education/to_sparse_input/indicesWherePlinear/linear_model/linear_model/linear_model/education/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Nlinear/linear_model/linear_model/linear_model/education/to_sparse_input/valuesGatherNdBlinear/linear_model/linear_model/linear_model/education/ExpandDimsOlinear/linear_model/linear_model/linear_model/education/to_sparse_input/indices*
Tparams0*#
_output_shapes
:?????????*
Tindices0	
?
Slinear/linear_model/linear_model/linear_model/education/to_sparse_input/dense_shapeShapeBlinear/linear_model/linear_model/linear_model/education/ExpandDims*
T0*
out_type0	*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/education/education_lookup/ConstConst*
dtype0*
_output_shapes
:*?
value?B?B
 BachelorsB HS-gradB 11thB MastersB 9thB Some-collegeB Assoc-acdmB
 Assoc-vocB 7th-8thB
 DoctorateB Prof-schoolB 5th-6thB 10thB 1st-4thB
 PreschoolB 12th
?
Mlinear/linear_model/linear_model/linear_model/education/education_lookup/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
?
Tlinear/linear_model/linear_model/linear_model/education/education_lookup/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
?
Tlinear/linear_model/linear_model/linear_model/education/education_lookup/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
Nlinear/linear_model/linear_model/linear_model/education/education_lookup/rangeRangeTlinear/linear_model/linear_model/linear_model/education/education_lookup/range/startMlinear/linear_model/linear_model/linear_model/education/education_lookup/SizeTlinear/linear_model/linear_model/linear_model/education/education_lookup/range/delta*
_output_shapes
:
?
Mlinear/linear_model/linear_model/linear_model/education/education_lookup/CastCastNlinear/linear_model/linear_model/linear_model/education/education_lookup/range*
_output_shapes
:*

DstT0	*

SrcT0
?
Ylinear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/ConstConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
^linear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/hash_tableHashTableV2*
	key_dtype0*
value_dtype0	*
_output_shapes
: 
?
rlinear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2^linear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/hash_tableNlinear/linear_model/linear_model/linear_model/education/education_lookup/ConstMlinear/linear_model/linear_model/linear_model/education/education_lookup/Cast*

Tout0	*	
Tin0
?
[linear/linear_model/linear_model/linear_model/education/hash_table_Lookup/LookupTableFindV2LookupTableFindV2^linear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/hash_tableNlinear/linear_model/linear_model/linear_model/education/to_sparse_input/valuesYlinear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/Const*#
_output_shapes
:?????????*	
Tin0*

Tout0	
?
Blinear/linear_model/linear_model/linear_model/education/Shape/CastCastSlinear/linear_model/linear_model/linear_model/education/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
?
Klinear/linear_model/linear_model/linear_model/education/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Mlinear/linear_model/linear_model/linear_model/education/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
?
Mlinear/linear_model/linear_model/linear_model/education/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Elinear/linear_model/linear_model/linear_model/education/strided_sliceStridedSliceBlinear/linear_model/linear_model/linear_model/education/Shape/CastKlinear/linear_model/linear_model/linear_model/education/strided_slice/stackMlinear/linear_model/linear_model/linear_model/education/strided_slice/stack_1Mlinear/linear_model/linear_model/linear_model/education/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
?
@linear/linear_model/linear_model/linear_model/education/Cast/x/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
>linear/linear_model/linear_model/linear_model/education/Cast/xPackElinear/linear_model/linear_model/linear_model/education/strided_slice@linear/linear_model/linear_model/linear_model/education/Cast/x/1*
N*
_output_shapes
:*
T0
?
<linear/linear_model/linear_model/linear_model/education/CastCast>linear/linear_model/linear_model/linear_model/education/Cast/x*

SrcT0*
_output_shapes
:*

DstT0	
?
Elinear/linear_model/linear_model/linear_model/education/SparseReshapeSparseReshapeOlinear/linear_model/linear_model/linear_model/education/to_sparse_input/indicesSlinear/linear_model/linear_model/linear_model/education/to_sparse_input/dense_shape<linear/linear_model/linear_model/linear_model/education/Cast*-
_output_shapes
:?????????:
?
Nlinear/linear_model/linear_model/linear_model/education/SparseReshape/IdentityIdentity[linear/linear_model/linear_model/linear_model/education/hash_table_Lookup/LookupTableFindV2*
T0	*#
_output_shapes
:?????????
?
Plinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/education/weighted_sum/SliceSliceGlinear/linear_model/linear_model/linear_model/education/SparseReshape:1Plinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice/beginOlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/education/weighted_sum/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
Ilinear/linear_model/linear_model/linear_model/education/weighted_sum/ProdProdJlinear/linear_model/linear_model/linear_model/education/weighted_sum/SliceJlinear/linear_model/linear_model/linear_model/education/weighted_sum/Const*
T0	*
_output_shapes
: 
?
Ulinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Mlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2GatherV2Glinear/linear_model/linear_model/linear_model/education/SparseReshape:1Ulinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2/indicesRlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
Klinear/linear_model/linear_model/linear_model/education/weighted_sum/Cast/xPackIlinear/linear_model/linear_model/linear_model/education/weighted_sum/ProdMlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2*
T0	*
N*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/SparseReshapeSparseReshapeElinear/linear_model/linear_model/linear_model/education/SparseReshapeGlinear/linear_model/linear_model/linear_model/education/SparseReshape:1Klinear/linear_model/linear_model/linear_model/education/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
[linear/linear_model/linear_model/linear_model/education/weighted_sum/SparseReshape/IdentityIdentityNlinear/linear_model/linear_model/linear_model/education/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
Slinear/linear_model/linear_model/linear_model/education/weighted_sum/GreaterEqual/yConst*
dtype0	*
_output_shapes
: *
value	B	 R 
?
Qlinear/linear_model/linear_model/linear_model/education/weighted_sum/GreaterEqualGreaterEqual[linear/linear_model/linear_model/linear_model/education/weighted_sum/SparseReshape/IdentitySlinear/linear_model/linear_model/linear_model/education/weighted_sum/GreaterEqual/y*#
_output_shapes
:?????????*
T0	
?
Jlinear/linear_model/linear_model/linear_model/education/weighted_sum/WhereWhereQlinear/linear_model/linear_model/linear_model/education/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Llinear/linear_model/linear_model/linear_model/education/weighted_sum/ReshapeReshapeJlinear/linear_model/linear_model/linear_model/education/weighted_sum/WhereRlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
Tlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Olinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_1GatherV2Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/SparseReshapeLlinear/linear_model/linear_model/linear_model/education/weighted_sum/ReshapeTlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_1/axis*'
_output_shapes
:?????????*
Taxis0*
Tindices0	*
Tparams0	
?
Tlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Olinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_2GatherV2[linear/linear_model/linear_model/linear_model/education/weighted_sum/SparseReshape/IdentityLlinear/linear_model/linear_model/linear_model/education/weighted_sum/ReshapeTlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_2/axis*#
_output_shapes
:?????????*
Taxis0*
Tindices0	*
Tparams0	
?
Mlinear/linear_model/linear_model/linear_model/education/weighted_sum/IdentityIdentityTlinear/linear_model/linear_model/linear_model/education/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
^linear/linear_model/linear_model/linear_model/education/weighted_sum/SparseFillEmptyRows/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
?
llinear/linear_model/linear_model/linear_model/education/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsOlinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_1Olinear/linear_model/linear_model/linear_model/education/weighted_sum/GatherV2_2Mlinear/linear_model/linear_model/linear_model/education/weighted_sum/Identity^linear/linear_model/linear_model/linear_model/education/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
plinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
?
rlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
rlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
?
jlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSlicellinear/linear_model/linear_model/linear_model/education/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsplinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_slice/stackrlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1rlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
Index0*
T0	*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:?????????
?
alinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/CastCastjlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:?????????*

DstT0
?
clinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/UniqueUniquenlinear/linear_model/linear_model/linear_model/education/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:?????????:?????????*
T0	
?
mlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather,linear/linear_model/education/weights/part_0clinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/Unique*
dtype0*'
_output_shapes
:?????????*
Tindices0	*?
_class5
31loc:@linear/linear_model/education/weights/part_0
?
vlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentitymlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*?
_class5
31loc:@linear/linear_model/education/weights/part_0*'
_output_shapes
:?????????
?
xlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identityvlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*'
_output_shapes
:?????????*
T0
?
\linear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparseSparseSegmentSumxlinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1elinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/Unique:1alinear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:?????????
?
Tlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"????   
?
Nlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape_1Reshapenlinear/linear_model/linear_model/linear_model/education/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Tlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
Jlinear/linear_model/linear_model/linear_model/education/weighted_sum/ShapeShape\linear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
Xlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
?
Zlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Zlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_sliceStridedSliceJlinear/linear_model/linear_model/linear_model/education/weighted_sum/ShapeXlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_slice/stackZlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_slice/stack_1Zlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Llinear/linear_model/linear_model/linear_model/education/weighted_sum/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
?
Jlinear/linear_model/linear_model/linear_model/education/weighted_sum/stackPackLlinear/linear_model/linear_model/linear_model/education/weighted_sum/stack/0Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/strided_slice*
N*
_output_shapes
:*
T0
?
Ilinear/linear_model/linear_model/linear_model/education/weighted_sum/TileTileNlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape_1Jlinear/linear_model/linear_model/linear_model/education/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
Olinear/linear_model/linear_model/linear_model/education/weighted_sum/zeros_like	ZerosLike\linear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Dlinear/linear_model/linear_model/linear_model/education/weighted_sumSelectIlinear/linear_model/linear_model/linear_model/education/weighted_sum/TileOlinear/linear_model/linear_model/linear_model/education/weighted_sum/zeros_like\linear/linear_model/linear_model/linear_model/education/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Klinear/linear_model/linear_model/linear_model/education/weighted_sum/Cast_1CastGlinear/linear_model/linear_model/linear_model/education/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
?
Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_1/sizeConst*
dtype0*
_output_shapes
:*
valueB:
?
Llinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_1SliceKlinear/linear_model/linear_model/linear_model/education/weighted_sum/Cast_1Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_1/beginQlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
Llinear/linear_model/linear_model/linear_model/education/weighted_sum/Shape_1ShapeDlinear/linear_model/linear_model/linear_model/education/weighted_sum*
T0*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?
Llinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_2SliceLlinear/linear_model/linear_model/linear_model/education/weighted_sum/Shape_1Rlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_2/beginQlinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
Plinear/linear_model/linear_model/linear_model/education/weighted_sum/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Klinear/linear_model/linear_model/linear_model/education/weighted_sum/concatConcatV2Llinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_1Llinear/linear_model/linear_model/linear_model/education/weighted_sum/Slice_2Plinear/linear_model/linear_model/linear_model/education/weighted_sum/concat/axis*
T0*
N*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape_2ReshapeDlinear/linear_model/linear_model/linear_model/education/weighted_sumKlinear/linear_model/linear_model/linear_model/education/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
Plinear/linear_model/linear_model/linear_model/education_X_occupation/SparseCrossSparseCrossBlinear/linear_model/linear_model/linear_model/education/ExpandDims]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_2*
num_buckets?N*
hashed_output(*
out_type0	*
N *<
_output_shapes*
(:?????????:?????????:*
dense_types
2*
hash_key?????*
sparse_types
 *
internal_type0
?
Olinear/linear_model/linear_model/linear_model/education_X_occupation/Shape/CastCastRlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseCross:2*

SrcT0	*
_output_shapes
:*

DstT0
?
Xlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
?
Zlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
?
Zlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_sliceStridedSliceOlinear/linear_model/linear_model/linear_model/education_X_occupation/Shape/CastXlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_slice/stackZlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_slice/stack_1Zlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
?
Mlinear/linear_model/linear_model/linear_model/education_X_occupation/Cast/x/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Klinear/linear_model/linear_model/linear_model/education_X_occupation/Cast/xPackRlinear/linear_model/linear_model/linear_model/education_X_occupation/strided_sliceMlinear/linear_model/linear_model/linear_model/education_X_occupation/Cast/x/1*
T0*
N*
_output_shapes
:
?
Ilinear/linear_model/linear_model/linear_model/education_X_occupation/CastCastKlinear/linear_model/linear_model/linear_model/education_X_occupation/Cast/x*

SrcT0*
_output_shapes
:*

DstT0	
?
Rlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshapeSparseReshapePlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseCrossRlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseCross:2Ilinear/linear_model/linear_model/linear_model/education_X_occupation/Cast*-
_output_shapes
:?????????:
?
[linear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshape/IdentityIdentityRlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseCross:1*
T0	*#
_output_shapes
:?????????
?
]linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice/sizeConst*
dtype0*
_output_shapes
:*
valueB:
?
Wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SliceSliceTlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshape:1]linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice/begin\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
Vlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/ProdProdWlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SliceWlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Const*
T0	*
_output_shapes
: 
?
blinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Zlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2GatherV2Tlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshape:1blinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2/indices_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Taxis0*
Tindices0
?
Xlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Cast/xPackVlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/ProdZlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2*
T0	*
N*
_output_shapes
:
?
_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseReshapeSparseReshapeRlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshapeTlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshape:1Xlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
hlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseReshape/IdentityIdentity[linear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
`linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
^linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GreaterEqualGreaterEqualhlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseReshape/Identity`linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GreaterEqual/y*#
_output_shapes
:?????????*
T0	
?
Wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/WhereWhere^linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/ReshapeReshapeWlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Where_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
alinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_1GatherV2_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseReshapeYlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshapealinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_1/axis*'
_output_shapes
:?????????*
Taxis0*
Tindices0	*
Tparams0	
?
alinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_2GatherV2hlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseReshape/IdentityYlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshapealinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_2/axis*#
_output_shapes
:?????????*
Taxis0*
Tindices0	*
Tparams0	
?
Zlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/IdentityIdentityalinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
klinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRows\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_1\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/GatherV2_2Zlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Identityklinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseFillEmptyRows/Const*T
_output_shapesB
@:?????????:?????????:?????????:?????????*
T0	
?
}linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
?
linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
?
wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows}linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stacklinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*

begin_mask*
end_mask*#
_output_shapes
:?????????*
T0	*
Index0*
shrink_axis_mask
?
nlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/CastCastwlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:?????????*

DstT0
?
plinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/UniqueUnique{linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:?????????:?????????*
T0	
?
zlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather9linear/linear_model/education_X_occupation/weights/part_0plinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0	*L
_classB
@>loc:@linear/linear_model/education_X_occupation/weights/part_0*
dtype0*'
_output_shapes
:?????????
?
?linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentityzlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*L
_classB
@>loc:@linear/linear_model/education_X_occupation/weights/part_0*'
_output_shapes
:?????????
?
?linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identity?linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
ilinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparseSparseSegmentSum?linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1rlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/Unique:1nlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:?????????
?
alinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"????   
?
[linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape_1Reshape{linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2alinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/ShapeShapeilinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
elinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
glinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
glinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_sliceStridedSliceWlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Shapeelinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_slice/stackglinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_slice/stack_1glinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
?
Ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
?
Wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/stackPackYlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/stack/0_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/strided_slice*
T0*
N*
_output_shapes
:
?
Vlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/TileTile[linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape_1Wlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/zeros_like	ZerosLikeilinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Qlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sumSelectVlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Tile\linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/zeros_likeilinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Xlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Cast_1CastTlinear/linear_model/linear_model/linear_model/education_X_occupation/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
?
_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
^linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_1SliceXlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Cast_1_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_1/begin^linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
Ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Shape_1ShapeQlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum*
_output_shapes
:*
T0
?
_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_2/beginConst*
dtype0*
_output_shapes
:*
valueB:
?
^linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?
Ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_2SliceYlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Shape_1_linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_2/begin^linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_2/size*
_output_shapes
:*
Index0*
T0
?
]linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Xlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/concatConcatV2Ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_1Ylinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Slice_2]linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/concat/axis*
N*
_output_shapes
:*
T0
?
[linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape_2ReshapeQlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sumXlinear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
Clinear/linear_model/linear_model/linear_model/gender/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
?linear/linear_model/linear_model/linear_model/gender/ExpandDims
ExpandDimsPlaceholderClinear/linear_model/linear_model/linear_model/gender/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
Slinear/linear_model/linear_model/linear_model/gender/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
Mlinear/linear_model/linear_model/linear_model/gender/to_sparse_input/NotEqualNotEqual?linear/linear_model/linear_model/linear_model/gender/ExpandDimsSlinear/linear_model/linear_model/linear_model/gender/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:?????????
?
Llinear/linear_model/linear_model/linear_model/gender/to_sparse_input/indicesWhereMlinear/linear_model/linear_model/linear_model/gender/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Klinear/linear_model/linear_model/linear_model/gender/to_sparse_input/valuesGatherNd?linear/linear_model/linear_model/linear_model/gender/ExpandDimsLlinear/linear_model/linear_model/linear_model/gender/to_sparse_input/indices*
Tparams0*#
_output_shapes
:?????????*
Tindices0	
?
Plinear/linear_model/linear_model/linear_model/gender/to_sparse_input/dense_shapeShape?linear/linear_model/linear_model/linear_model/gender/ExpandDims*
_output_shapes
:*
T0*
out_type0	
?
Hlinear/linear_model/linear_model/linear_model/gender/gender_lookup/ConstConst*#
valueBB FemaleB Male*
dtype0*
_output_shapes
:
?
Glinear/linear_model/linear_model/linear_model/gender/gender_lookup/SizeConst*
dtype0*
_output_shapes
: *
value	B :
?
Nlinear/linear_model/linear_model/linear_model/gender/gender_lookup/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
?
Nlinear/linear_model/linear_model/linear_model/gender/gender_lookup/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
Hlinear/linear_model/linear_model/linear_model/gender/gender_lookup/rangeRangeNlinear/linear_model/linear_model/linear_model/gender/gender_lookup/range/startGlinear/linear_model/linear_model/linear_model/gender/gender_lookup/SizeNlinear/linear_model/linear_model/linear_model/gender/gender_lookup/range/delta*
_output_shapes
:
?
Glinear/linear_model/linear_model/linear_model/gender/gender_lookup/CastCastHlinear/linear_model/linear_model/linear_model/gender/gender_lookup/range*
_output_shapes
:*

DstT0	*

SrcT0
?
Slinear/linear_model/linear_model/linear_model/gender/gender_lookup/hash_table/ConstConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
Xlinear/linear_model/linear_model/linear_model/gender/gender_lookup/hash_table/hash_tableHashTableV2*
value_dtype0	*
	key_dtype0*
_output_shapes
: 
?
llinear/linear_model/linear_model/linear_model/gender/gender_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2Xlinear/linear_model/linear_model/linear_model/gender/gender_lookup/hash_table/hash_tableHlinear/linear_model/linear_model/linear_model/gender/gender_lookup/ConstGlinear/linear_model/linear_model/linear_model/gender/gender_lookup/Cast*	
Tin0*

Tout0	
?
Xlinear/linear_model/linear_model/linear_model/gender/hash_table_Lookup/LookupTableFindV2LookupTableFindV2Xlinear/linear_model/linear_model/linear_model/gender/gender_lookup/hash_table/hash_tableKlinear/linear_model/linear_model/linear_model/gender/to_sparse_input/valuesSlinear/linear_model/linear_model/linear_model/gender/gender_lookup/hash_table/Const*

Tout0	*#
_output_shapes
:?????????*	
Tin0
?
?linear/linear_model/linear_model/linear_model/gender/Shape/CastCastPlinear/linear_model/linear_model/linear_model/gender/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
?
Hlinear/linear_model/linear_model/linear_model/gender/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/gender/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/gender/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Blinear/linear_model/linear_model/linear_model/gender/strided_sliceStridedSlice?linear/linear_model/linear_model/linear_model/gender/Shape/CastHlinear/linear_model/linear_model/linear_model/gender/strided_slice/stackJlinear/linear_model/linear_model/linear_model/gender/strided_slice/stack_1Jlinear/linear_model/linear_model/linear_model/gender/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
?
=linear/linear_model/linear_model/linear_model/gender/Cast/x/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
;linear/linear_model/linear_model/linear_model/gender/Cast/xPackBlinear/linear_model/linear_model/linear_model/gender/strided_slice=linear/linear_model/linear_model/linear_model/gender/Cast/x/1*
T0*
N*
_output_shapes
:
?
9linear/linear_model/linear_model/linear_model/gender/CastCast;linear/linear_model/linear_model/linear_model/gender/Cast/x*
_output_shapes
:*

DstT0	*

SrcT0
?
Blinear/linear_model/linear_model/linear_model/gender/SparseReshapeSparseReshapeLlinear/linear_model/linear_model/linear_model/gender/to_sparse_input/indicesPlinear/linear_model/linear_model/linear_model/gender/to_sparse_input/dense_shape9linear/linear_model/linear_model/linear_model/gender/Cast*-
_output_shapes
:?????????:
?
Klinear/linear_model/linear_model/linear_model/gender/SparseReshape/IdentityIdentityXlinear/linear_model/linear_model/linear_model/gender/hash_table_Lookup/LookupTableFindV2*
T0	*#
_output_shapes
:?????????
?
Mlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
Llinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Glinear/linear_model/linear_model/linear_model/gender/weighted_sum/SliceSliceDlinear/linear_model/linear_model/linear_model/gender/SparseReshape:1Mlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice/beginLlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Glinear/linear_model/linear_model/linear_model/gender/weighted_sum/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
?
Flinear/linear_model/linear_model/linear_model/gender/weighted_sum/ProdProdGlinear/linear_model/linear_model/linear_model/gender/weighted_sum/SliceGlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Const*
T0	*
_output_shapes
: 
?
Rlinear/linear_model/linear_model/linear_model/gender/weighted_sum/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
Olinear/linear_model/linear_model/linear_model/gender/weighted_sum/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
Jlinear/linear_model/linear_model/linear_model/gender/weighted_sum/GatherV2GatherV2Dlinear/linear_model/linear_model/linear_model/gender/SparseReshape:1Rlinear/linear_model/linear_model/linear_model/gender/weighted_sum/GatherV2/indicesOlinear/linear_model/linear_model/linear_model/gender/weighted_sum/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Taxis0*
Tindices0
?
Hlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Cast/xPackFlinear/linear_model/linear_model/linear_model/gender/weighted_sum/ProdJlinear/linear_model/linear_model/linear_model/gender/weighted_sum/GatherV2*
T0	*
N*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/gender/weighted_sum/SparseReshapeSparseReshapeBlinear/linear_model/linear_model/linear_model/gender/SparseReshapeDlinear/linear_model/linear_model/linear_model/gender/SparseReshape:1Hlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
Xlinear/linear_model/linear_model/linear_model/gender/weighted_sum/SparseReshape/IdentityIdentityKlinear/linear_model/linear_model/linear_model/gender/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
Plinear/linear_model/linear_model/linear_model/gender/weighted_sum/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
Nlinear/linear_model/linear_model/linear_model/gender/weighted_sum/GreaterEqualGreaterEqualXlinear/linear_model/linear_model/linear_model/gender/weighted_sum/SparseReshape/IdentityPlinear/linear_model/linear_model/linear_model/gender/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
Glinear/linear_model/linear_model/linear_model/gender/weighted_sum/WhereWhereNlinear/linear_model/linear_model/linear_model/gender/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
Olinear/linear_model/linear_model/linear_model/gender/weighted_sum/Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Ilinear/linear_model/linear_model/linear_model/gender/weighted_sum/ReshapeReshapeGlinear/linear_model/linear_model/linear_model/gender/weighted_sum/WhereOlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Reshape/shape*#
_output_shapes
:?????????*
T0	
?
Qlinear/linear_model/linear_model/linear_model/gender/weighted_sum/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Llinear/linear_model/linear_model/linear_model/gender/weighted_sum/GatherV2_1GatherV2Olinear/linear_model/linear_model/linear_model/gender/weighted_sum/SparseReshapeIlinear/linear_model/linear_model/linear_model/gender/weighted_sum/ReshapeQlinear/linear_model/linear_model/linear_model/gender/weighted_sum/GatherV2_1/axis*
Tparams0	*'
_output_shapes
:?????????*
Taxis0*
Tindices0	
?
Qlinear/linear_model/linear_model/linear_model/gender/weighted_sum/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
Llinear/linear_model/linear_model/linear_model/gender/weighted_sum/GatherV2_2GatherV2Xlinear/linear_model/linear_model/linear_model/gender/weighted_sum/SparseReshape/IdentityIlinear/linear_model/linear_model/linear_model/gender/weighted_sum/ReshapeQlinear/linear_model/linear_model/linear_model/gender/weighted_sum/GatherV2_2/axis*
Tparams0	*#
_output_shapes
:?????????*
Taxis0*
Tindices0	
?
Jlinear/linear_model/linear_model/linear_model/gender/weighted_sum/IdentityIdentityQlinear/linear_model/linear_model/linear_model/gender/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
[linear/linear_model/linear_model/linear_model/gender/weighted_sum/SparseFillEmptyRows/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
?
ilinear/linear_model/linear_model/linear_model/gender/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsLlinear/linear_model/linear_model/linear_model/gender/weighted_sum/GatherV2_1Llinear/linear_model/linear_model/linear_model/gender/weighted_sum/GatherV2_2Jlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Identity[linear/linear_model/linear_model/linear_model/gender/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
mlinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
?
olinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
olinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
?
glinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceilinear/linear_model/linear_model/linear_model/gender/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsmlinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_slice/stackolinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1olinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*

begin_mask*
end_mask*#
_output_shapes
:?????????*
Index0*
T0	*
shrink_axis_mask
?
^linear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/CastCastglinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/strided_slice*#
_output_shapes
:?????????*

DstT0*

SrcT0	
?
`linear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/UniqueUniqueklinear/linear_model/linear_model/linear_model/gender/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
jlinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather)linear/linear_model/gender/weights/part_0`linear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0	*<
_class2
0.loc:@linear/linear_model/gender/weights/part_0*
dtype0*'
_output_shapes
:?????????
?
slinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentityjlinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:?????????*
T0*<
_class2
0.loc:@linear/linear_model/gender/weights/part_0
?
ulinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identityslinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
Ylinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparseSparseSegmentSumulinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1blinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/Unique:1^linear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse/Cast*'
_output_shapes
:?????????*
T0
?
Qlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Reshape_1/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
?
Klinear/linear_model/linear_model/linear_model/gender/weighted_sum/Reshape_1Reshapeklinear/linear_model/linear_model/linear_model/gender/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Qlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
Glinear/linear_model/linear_model/linear_model/gender/weighted_sum/ShapeShapeYlinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
Ulinear/linear_model/linear_model/linear_model/gender/weighted_sum/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/gender/weighted_sum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/gender/weighted_sum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/gender/weighted_sum/strided_sliceStridedSliceGlinear/linear_model/linear_model/linear_model/gender/weighted_sum/ShapeUlinear/linear_model/linear_model/linear_model/gender/weighted_sum/strided_slice/stackWlinear/linear_model/linear_model/linear_model/gender/weighted_sum/strided_slice/stack_1Wlinear/linear_model/linear_model/linear_model/gender/weighted_sum/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
?
Ilinear/linear_model/linear_model/linear_model/gender/weighted_sum/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
?
Glinear/linear_model/linear_model/linear_model/gender/weighted_sum/stackPackIlinear/linear_model/linear_model/linear_model/gender/weighted_sum/stack/0Olinear/linear_model/linear_model/linear_model/gender/weighted_sum/strided_slice*
T0*
N*
_output_shapes
:
?
Flinear/linear_model/linear_model/linear_model/gender/weighted_sum/TileTileKlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Reshape_1Glinear/linear_model/linear_model/linear_model/gender/weighted_sum/stack*0
_output_shapes
:??????????????????*
T0

?
Llinear/linear_model/linear_model/linear_model/gender/weighted_sum/zeros_like	ZerosLikeYlinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Alinear/linear_model/linear_model/linear_model/gender/weighted_sumSelectFlinear/linear_model/linear_model/linear_model/gender/weighted_sum/TileLlinear/linear_model/linear_model/linear_model/gender/weighted_sum/zeros_likeYlinear/linear_model/linear_model/linear_model/gender/weighted_sum/embedding_lookup_sparse*'
_output_shapes
:?????????*
T0
?
Hlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Cast_1CastDlinear/linear_model/linear_model/linear_model/gender/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
?
Olinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Ilinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice_1SliceHlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Cast_1Olinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice_1/beginNlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
Ilinear/linear_model/linear_model/linear_model/gender/weighted_sum/Shape_1ShapeAlinear/linear_model/linear_model/linear_model/gender/weighted_sum*
T0*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice_2/sizeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Ilinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice_2SliceIlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Shape_1Olinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice_2/beginNlinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice_2/size*
_output_shapes
:*
Index0*
T0
?
Mlinear/linear_model/linear_model/linear_model/gender/weighted_sum/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Hlinear/linear_model/linear_model/linear_model/gender/weighted_sum/concatConcatV2Ilinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice_1Ilinear/linear_model/linear_model/linear_model/gender/weighted_sum/Slice_2Mlinear/linear_model/linear_model/linear_model/gender/weighted_sum/concat/axis*
N*
_output_shapes
:*
T0
?
Klinear/linear_model/linear_model/linear_model/gender/weighted_sum/Reshape_2ReshapeAlinear/linear_model/linear_model/linear_model/gender/weighted_sumHlinear/linear_model/linear_model/linear_model/gender/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
Klinear/linear_model/linear_model/linear_model/marital_status/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Glinear/linear_model/linear_model/linear_model/marital_status/ExpandDims
ExpandDimsPlaceholder_3Klinear/linear_model/linear_model/linear_model/marital_status/ExpandDims/dim*'
_output_shapes
:?????????*
T0
?
[linear/linear_model/linear_model/linear_model/marital_status/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
Ulinear/linear_model/linear_model/linear_model/marital_status/to_sparse_input/NotEqualNotEqualGlinear/linear_model/linear_model/linear_model/marital_status/ExpandDims[linear/linear_model/linear_model/linear_model/marital_status/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:?????????
?
Tlinear/linear_model/linear_model/linear_model/marital_status/to_sparse_input/indicesWhereUlinear/linear_model/linear_model/linear_model/marital_status/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Slinear/linear_model/linear_model/linear_model/marital_status/to_sparse_input/valuesGatherNdGlinear/linear_model/linear_model/linear_model/marital_status/ExpandDimsTlinear/linear_model/linear_model/linear_model/marital_status/to_sparse_input/indices*
Tparams0*#
_output_shapes
:?????????*
Tindices0	
?
Xlinear/linear_model/linear_model/linear_model/marital_status/to_sparse_input/dense_shapeShapeGlinear/linear_model/linear_model/linear_model/marital_status/ExpandDims*
T0*
out_type0	*
_output_shapes
:
?
Xlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/ConstConst*
dtype0*
_output_shapes
:*?
value|BzB Married-civ-spouseB	 DivorcedB Married-spouse-absentB Never-marriedB
 SeparatedB Married-AF-spouseB Widowed
?
Wlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
?
^linear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
?
^linear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
Xlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/rangeRange^linear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/range/startWlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/Size^linear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/range/delta*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/CastCastXlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/range*
_output_shapes
:*

DstT0	*

SrcT0
?
clinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/ConstConst*
valueB	 R
?????????*
dtype0	*
_output_shapes
: 
?
hlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/hash_tableHashTableV2*
value_dtype0	*
	key_dtype0*
_output_shapes
: 
?
|linear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2hlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/hash_tableXlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/ConstWlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/Cast*	
Tin0*

Tout0	
?
`linear/linear_model/linear_model/linear_model/marital_status/hash_table_Lookup/LookupTableFindV2LookupTableFindV2hlinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/hash_tableSlinear/linear_model/linear_model/linear_model/marital_status/to_sparse_input/valuesclinear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/Const*

Tout0	*#
_output_shapes
:?????????*	
Tin0
?
Glinear/linear_model/linear_model/linear_model/marital_status/Shape/CastCastXlinear/linear_model/linear_model/linear_model/marital_status/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
?
Plinear/linear_model/linear_model/linear_model/marital_status/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/marital_status/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/marital_status/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/marital_status/strided_sliceStridedSliceGlinear/linear_model/linear_model/linear_model/marital_status/Shape/CastPlinear/linear_model/linear_model/linear_model/marital_status/strided_slice/stackRlinear/linear_model/linear_model/linear_model/marital_status/strided_slice/stack_1Rlinear/linear_model/linear_model/linear_model/marital_status/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
?
Elinear/linear_model/linear_model/linear_model/marital_status/Cast/x/1Const*
dtype0*
_output_shapes
: *
valueB :
?????????
?
Clinear/linear_model/linear_model/linear_model/marital_status/Cast/xPackJlinear/linear_model/linear_model/linear_model/marital_status/strided_sliceElinear/linear_model/linear_model/linear_model/marital_status/Cast/x/1*
T0*
N*
_output_shapes
:
?
Alinear/linear_model/linear_model/linear_model/marital_status/CastCastClinear/linear_model/linear_model/linear_model/marital_status/Cast/x*
_output_shapes
:*

DstT0	*

SrcT0
?
Jlinear/linear_model/linear_model/linear_model/marital_status/SparseReshapeSparseReshapeTlinear/linear_model/linear_model/linear_model/marital_status/to_sparse_input/indicesXlinear/linear_model/linear_model/linear_model/marital_status/to_sparse_input/dense_shapeAlinear/linear_model/linear_model/linear_model/marital_status/Cast*-
_output_shapes
:?????????:
?
Slinear/linear_model/linear_model/linear_model/marital_status/SparseReshape/IdentityIdentity`linear/linear_model/linear_model/linear_model/marital_status/hash_table_Lookup/LookupTableFindV2*
T0	*#
_output_shapes
:?????????
?
Ulinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
Tlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SliceSliceLlinear/linear_model/linear_model/linear_model/marital_status/SparseReshape:1Ulinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice/beginTlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/ProdProdOlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SliceOlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Const*
T0	*
_output_shapes
: 
?
Zlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :
?
Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
Rlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2GatherV2Llinear/linear_model/linear_model/linear_model/marital_status/SparseReshape:1Zlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2/indicesWlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Taxis0*
Tindices0
?
Plinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Cast/xPackNlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/ProdRlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2*
T0	*
N*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseReshapeSparseReshapeJlinear/linear_model/linear_model/linear_model/marital_status/SparseReshapeLlinear/linear_model/linear_model/linear_model/marital_status/SparseReshape:1Plinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
`linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseReshape/IdentityIdentitySlinear/linear_model/linear_model/linear_model/marital_status/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
Xlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
Vlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GreaterEqualGreaterEqual`linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseReshape/IdentityXlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
Olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/WhereWhereVlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/ReshapeReshapeOlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/WhereWlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
Ylinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Tlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_1GatherV2Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseReshapeQlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/ReshapeYlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_1/axis*
Tindices0	*
Tparams0	*'
_output_shapes
:?????????*
Taxis0
?
Ylinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Tlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_2GatherV2`linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseReshape/IdentityQlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/ReshapeYlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_2/axis*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????*
Taxis0
?
Rlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/IdentityIdentityYlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseReshape:1*
_output_shapes
:*
T0	
?
clinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsTlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_1Tlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/GatherV2_2Rlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Identityclinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
ulinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
?
wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
?
olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceqlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsulinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_slice/stackwlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
T0	*
Index0*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:?????????
?
flinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/CastCastolinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:?????????*

DstT0
?
hlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/UniqueUniqueslinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
rlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather1linear/linear_model/marital_status/weights/part_0hlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0	*D
_class:
86loc:@linear/linear_model/marital_status/weights/part_0*
dtype0*'
_output_shapes
:?????????
?
{linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentityrlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*D
_class:
86loc:@linear/linear_model/marital_status/weights/part_0*'
_output_shapes
:?????????
?
}linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identity{linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
alinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparseSparseSegmentSum}linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1jlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/Unique:1flinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse/Cast*'
_output_shapes
:?????????*
T0
?
Ylinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape_1/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape_1Reshapeslinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Ylinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
Olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/ShapeShapealinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
]linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
_linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
?
_linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_sliceStridedSliceOlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Shape]linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_slice/stack_linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_slice/stack_1_linear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
?
Qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
?
Olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/stackPackQlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/stack/0Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/strided_slice*
T0*
N*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/TileTileSlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape_1Olinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/stack*0
_output_shapes
:??????????????????*
T0

?
Tlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/zeros_like	ZerosLikealinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Ilinear/linear_model/linear_model/linear_model/marital_status/weighted_sumSelectNlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/TileTlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/zeros_likealinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Plinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Cast_1CastLlinear/linear_model/linear_model/linear_model/marital_status/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
?
Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
Vlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_1SlicePlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Cast_1Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_1/beginVlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Shape_1ShapeIlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum*
_output_shapes
:*
T0
?
Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
?
Vlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_2/sizeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_2SliceQlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Shape_1Wlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_2/beginVlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_2/size*
_output_shapes
:*
Index0*
T0
?
Ulinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Plinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/concatConcatV2Qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_1Qlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Slice_2Ulinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/concat/axis*
T0*
N*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape_2ReshapeIlinear/linear_model/linear_model/linear_model/marital_status/weighted_sumPlinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
Klinear/linear_model/linear_model/linear_model/native_country/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
?????????
?
Glinear/linear_model/linear_model/linear_model/native_country/ExpandDims
ExpandDimsPlaceholder_7Klinear/linear_model/linear_model/linear_model/native_country/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
[linear/linear_model/linear_model/linear_model/native_country/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
Ulinear/linear_model/linear_model/linear_model/native_country/to_sparse_input/NotEqualNotEqualGlinear/linear_model/linear_model/linear_model/native_country/ExpandDims[linear/linear_model/linear_model/linear_model/native_country/to_sparse_input/ignore_value/x*'
_output_shapes
:?????????*
T0
?
Tlinear/linear_model/linear_model/linear_model/native_country/to_sparse_input/indicesWhereUlinear/linear_model/linear_model/linear_model/native_country/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Slinear/linear_model/linear_model/linear_model/native_country/to_sparse_input/valuesGatherNdGlinear/linear_model/linear_model/linear_model/native_country/ExpandDimsTlinear/linear_model/linear_model/linear_model/native_country/to_sparse_input/indices*
Tparams0*#
_output_shapes
:?????????*
Tindices0	
?
Xlinear/linear_model/linear_model/linear_model/native_country/to_sparse_input/dense_shapeShapeGlinear/linear_model/linear_model/linear_model/native_country/ExpandDims*
_output_shapes
:*
T0*
out_type0	
?
Clinear/linear_model/linear_model/linear_model/native_country/lookupStringToHashBucketFastSlinear/linear_model/linear_model/linear_model/native_country/to_sparse_input/values*
num_bucketsd*#
_output_shapes
:?????????
?
Glinear/linear_model/linear_model/linear_model/native_country/Shape/CastCastXlinear/linear_model/linear_model/linear_model/native_country/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
?
Plinear/linear_model/linear_model/linear_model/native_country/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/native_country/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/native_country/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/native_country/strided_sliceStridedSliceGlinear/linear_model/linear_model/linear_model/native_country/Shape/CastPlinear/linear_model/linear_model/linear_model/native_country/strided_slice/stackRlinear/linear_model/linear_model/linear_model/native_country/strided_slice/stack_1Rlinear/linear_model/linear_model/linear_model/native_country/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Elinear/linear_model/linear_model/linear_model/native_country/Cast/x/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Clinear/linear_model/linear_model/linear_model/native_country/Cast/xPackJlinear/linear_model/linear_model/linear_model/native_country/strided_sliceElinear/linear_model/linear_model/linear_model/native_country/Cast/x/1*
T0*
N*
_output_shapes
:
?
Alinear/linear_model/linear_model/linear_model/native_country/CastCastClinear/linear_model/linear_model/linear_model/native_country/Cast/x*

SrcT0*
_output_shapes
:*

DstT0	
?
Jlinear/linear_model/linear_model/linear_model/native_country/SparseReshapeSparseReshapeTlinear/linear_model/linear_model/linear_model/native_country/to_sparse_input/indicesXlinear/linear_model/linear_model/linear_model/native_country/to_sparse_input/dense_shapeAlinear/linear_model/linear_model/linear_model/native_country/Cast*-
_output_shapes
:?????????:
?
Slinear/linear_model/linear_model/linear_model/native_country/SparseReshape/IdentityIdentityClinear/linear_model/linear_model/linear_model/native_country/lookup*#
_output_shapes
:?????????*
T0	
?
Ulinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
Tlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/native_country/weighted_sum/SliceSliceLlinear/linear_model/linear_model/linear_model/native_country/SparseReshape:1Ulinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice/beginTlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice/size*
_output_shapes
:*
Index0*
T0	
?
Olinear/linear_model/linear_model/linear_model/native_country/weighted_sum/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
?
Nlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/ProdProdOlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/SliceOlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Const*
_output_shapes
: *
T0	
?
Zlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :
?
Wlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Rlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GatherV2GatherV2Llinear/linear_model/linear_model/linear_model/native_country/SparseReshape:1Zlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GatherV2/indicesWlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GatherV2/axis*
Tindices0*
Tparams0	*
_output_shapes
: *
Taxis0
?
Plinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Cast/xPackNlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/ProdRlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GatherV2*
T0	*
N*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/SparseReshapeSparseReshapeJlinear/linear_model/linear_model/linear_model/native_country/SparseReshapeLlinear/linear_model/linear_model/linear_model/native_country/SparseReshape:1Plinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
`linear/linear_model/linear_model/linear_model/native_country/weighted_sum/SparseReshape/IdentityIdentitySlinear/linear_model/linear_model/linear_model/native_country/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
Xlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
Vlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GreaterEqualGreaterEqual`linear/linear_model/linear_model/linear_model/native_country/weighted_sum/SparseReshape/IdentityXlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
Olinear/linear_model/linear_model/linear_model/native_country/weighted_sum/WhereWhereVlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?
Qlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/ReshapeReshapeOlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/WhereWlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
Ylinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
Tlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GatherV2_1GatherV2Wlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/SparseReshapeQlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/ReshapeYlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GatherV2_1/axis*'
_output_shapes
:?????????*
Taxis0*
Tindices0	*
Tparams0	
?
Ylinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
Tlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GatherV2_2GatherV2`linear/linear_model/linear_model/linear_model/native_country/weighted_sum/SparseReshape/IdentityQlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/ReshapeYlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GatherV2_2/axis*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????*
Taxis0
?
Rlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/IdentityIdentityYlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
clinear/linear_model/linear_model/linear_model/native_country/weighted_sum/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
qlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsTlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GatherV2_1Tlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/GatherV2_2Rlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Identityclinear/linear_model/linear_model/linear_model/native_country/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
ulinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
?
wlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
wlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
?
olinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceqlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsulinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/strided_slice/stackwlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1wlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
T0	*
Index0*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:?????????
?
flinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/CastCastolinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:?????????*

DstT0
?
hlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/UniqueUniqueslinear/linear_model/linear_model/linear_model/native_country/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
rlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather1linear/linear_model/native_country/weights/part_0hlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/Unique*
dtype0*'
_output_shapes
:?????????*
Tindices0	*D
_class:
86loc:@linear/linear_model/native_country/weights/part_0
?
{linear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentityrlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*D
_class:
86loc:@linear/linear_model/native_country/weights/part_0*'
_output_shapes
:?????????
?
}linear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identity{linear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
alinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparseSparseSegmentSum}linear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1jlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/Unique:1flinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:?????????
?
Ylinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Reshape_1/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Reshape_1Reshapeslinear/linear_model/linear_model/linear_model/native_country/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Ylinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
Olinear/linear_model/linear_model/linear_model/native_country/weighted_sum/ShapeShapealinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse*
_output_shapes
:*
T0
?
]linear/linear_model/linear_model/linear_model/native_country/weighted_sum/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
_linear/linear_model/linear_model/linear_model/native_country/weighted_sum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
_linear/linear_model/linear_model/linear_model/native_country/weighted_sum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/strided_sliceStridedSliceOlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Shape]linear/linear_model/linear_model/linear_model/native_country/weighted_sum/strided_slice/stack_linear/linear_model/linear_model/linear_model/native_country/weighted_sum/strided_slice/stack_1_linear/linear_model/linear_model/linear_model/native_country/weighted_sum/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
?
Qlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
?
Olinear/linear_model/linear_model/linear_model/native_country/weighted_sum/stackPackQlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/stack/0Wlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/strided_slice*
T0*
N*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/TileTileSlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Reshape_1Olinear/linear_model/linear_model/linear_model/native_country/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
Tlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/zeros_like	ZerosLikealinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Ilinear/linear_model/linear_model/linear_model/native_country/weighted_sumSelectNlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/TileTlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/zeros_likealinear/linear_model/linear_model/linear_model/native_country/weighted_sum/embedding_lookup_sparse*'
_output_shapes
:?????????*
T0
?
Plinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Cast_1CastLlinear/linear_model/linear_model/linear_model/native_country/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
?
Wlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
Vlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice_1SlicePlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Cast_1Wlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice_1/beginVlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice_1/size*
_output_shapes
:*
Index0*
T0
?
Qlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Shape_1ShapeIlinear/linear_model/linear_model/linear_model/native_country/weighted_sum*
T0*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
?
Vlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice_2/sizeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice_2SliceQlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Shape_1Wlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice_2/beginVlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
Ulinear/linear_model/linear_model/linear_model/native_country/weighted_sum/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Plinear/linear_model/linear_model/linear_model/native_country/weighted_sum/concatConcatV2Qlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice_1Qlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Slice_2Ulinear/linear_model/linear_model/linear_model/native_country/weighted_sum/concat/axis*
T0*
N*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Reshape_2ReshapeIlinear/linear_model/linear_model/linear_model/native_country/weighted_sumPlinear/linear_model/linear_model/linear_model/native_country/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
Ulinear/linear_model/linear_model/linear_model/native_country_X_occupation/SparseCrossSparseCrossGlinear/linear_model/linear_model/linear_model/native_country/ExpandDims]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_2*
num_buckets?N*
hashed_output(*
out_type0	*
N *
dense_types
2*<
_output_shapes*
(:?????????:?????????:*
hash_key?????*
sparse_types
 *
internal_type0
?
Tlinear/linear_model/linear_model/linear_model/native_country_X_occupation/Shape/CastCastWlinear/linear_model/linear_model/linear_model/native_country_X_occupation/SparseCross:2*

SrcT0	*
_output_shapes
:*

DstT0
?
]linear/linear_model/linear_model/linear_model/native_country_X_occupation/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
?
_linear/linear_model/linear_model/linear_model/native_country_X_occupation/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
_linear/linear_model/linear_model/linear_model/native_country_X_occupation/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Wlinear/linear_model/linear_model/linear_model/native_country_X_occupation/strided_sliceStridedSliceTlinear/linear_model/linear_model/linear_model/native_country_X_occupation/Shape/Cast]linear/linear_model/linear_model/linear_model/native_country_X_occupation/strided_slice/stack_linear/linear_model/linear_model/linear_model/native_country_X_occupation/strided_slice/stack_1_linear/linear_model/linear_model/linear_model/native_country_X_occupation/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
?
Rlinear/linear_model/linear_model/linear_model/native_country_X_occupation/Cast/x/1Const*
dtype0*
_output_shapes
: *
valueB :
?????????
?
Plinear/linear_model/linear_model/linear_model/native_country_X_occupation/Cast/xPackWlinear/linear_model/linear_model/linear_model/native_country_X_occupation/strided_sliceRlinear/linear_model/linear_model/linear_model/native_country_X_occupation/Cast/x/1*
T0*
N*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/native_country_X_occupation/CastCastPlinear/linear_model/linear_model/linear_model/native_country_X_occupation/Cast/x*

SrcT0*
_output_shapes
:*

DstT0	
?
Wlinear/linear_model/linear_model/linear_model/native_country_X_occupation/SparseReshapeSparseReshapeUlinear/linear_model/linear_model/linear_model/native_country_X_occupation/SparseCrossWlinear/linear_model/linear_model/linear_model/native_country_X_occupation/SparseCross:2Nlinear/linear_model/linear_model/linear_model/native_country_X_occupation/Cast*-
_output_shapes
:?????????:
?
`linear/linear_model/linear_model/linear_model/native_country_X_occupation/SparseReshape/IdentityIdentityWlinear/linear_model/linear_model/linear_model/native_country_X_occupation/SparseCross:1*
T0	*#
_output_shapes
:?????????
?
blinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
alinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
\linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/SliceSliceYlinear/linear_model/linear_model/linear_model/native_country_X_occupation/SparseReshape:1blinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice/beginalinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
\linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
[linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/ProdProd\linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice\linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Const*
T0	*
_output_shapes
: 
?
glinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :
?
dlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
_linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GatherV2GatherV2Ylinear/linear_model/linear_model/linear_model/native_country_X_occupation/SparseReshape:1glinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GatherV2/indicesdlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GatherV2/axis*
_output_shapes
: *
Taxis0*
Tindices0*
Tparams0	
?
]linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Cast/xPack[linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Prod_linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GatherV2*
T0	*
N*
_output_shapes
:
?
dlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/SparseReshapeSparseReshapeWlinear/linear_model/linear_model/linear_model/native_country_X_occupation/SparseReshapeYlinear/linear_model/linear_model/linear_model/native_country_X_occupation/SparseReshape:1]linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
mlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/SparseReshape/IdentityIdentity`linear/linear_model/linear_model/linear_model/native_country_X_occupation/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
elinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
clinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GreaterEqualGreaterEqualmlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/SparseReshape/Identityelinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
\linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/WhereWhereclinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
dlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?
^linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/ReshapeReshape\linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Wheredlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
flinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
alinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GatherV2_1GatherV2dlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/SparseReshape^linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Reshapeflinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GatherV2_1/axis*
Tparams0	*'
_output_shapes
:?????????*
Taxis0*
Tindices0	
?
flinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GatherV2_2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
alinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GatherV2_2GatherV2mlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/SparseReshape/Identity^linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Reshapeflinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GatherV2_2/axis*#
_output_shapes
:?????????*
Taxis0*
Tindices0	*
Tparams0	
?
_linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/IdentityIdentityflinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
plinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
~linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsalinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GatherV2_1alinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/GatherV2_2_linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Identityplinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
?linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB"        
?
?linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
?linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
?
|linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSlice~linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows?linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack?linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1?linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:?????????*
Index0*
T0	
?
slinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/CastCast|linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:?????????*

DstT0
?
ulinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/UniqueUnique?linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather>linear/linear_model/native_country_X_occupation/weights/part_0ulinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/Unique*
dtype0*'
_output_shapes
:?????????*
Tindices0	*Q
_classG
ECloc:@linear/linear_model/native_country_X_occupation/weights/part_0
?
?linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentitylinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup*'
_output_shapes
:?????????*
T0*Q
_classG
ECloc:@linear/linear_model/native_country_X_occupation/weights/part_0
?
?linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identity?linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
nlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparseSparseSegmentSum?linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1wlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/Unique:1slinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:?????????
?
flinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Reshape_1/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
?
`linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Reshape_1Reshape?linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2flinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
\linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/ShapeShapenlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse*
_output_shapes
:*
T0
?
jlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
llinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
llinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
dlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/strided_sliceStridedSlice\linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Shapejlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/strided_slice/stackllinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/strided_slice/stack_1llinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
?
^linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
?
\linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/stackPack^linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/stack/0dlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/strided_slice*
T0*
N*
_output_shapes
:
?
[linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/TileTile`linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Reshape_1\linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/stack*0
_output_shapes
:??????????????????*
T0

?
alinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/zeros_like	ZerosLikenlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Vlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sumSelect[linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Tilealinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/zeros_likenlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/embedding_lookup_sparse*'
_output_shapes
:?????????*
T0
?
]linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Cast_1CastYlinear/linear_model/linear_model/linear_model/native_country_X_occupation/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
?
dlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 
?
clinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
^linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice_1Slice]linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Cast_1dlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice_1/beginclinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
^linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Shape_1ShapeVlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum*
T0*
_output_shapes
:
?
dlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
?
clinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice_2/sizeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
^linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice_2Slice^linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Shape_1dlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice_2/beginclinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
blinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
]linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/concatConcatV2^linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice_1^linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Slice_2blinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/concat/axis*
N*
_output_shapes
:*
T0
?
`linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Reshape_2ReshapeVlinear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum]linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/occupation/to_sparse_input/ignore_value/xConst*
dtype0*
_output_shapes
: *
valueB B 
?
Qlinear/linear_model/linear_model/linear_model/occupation/to_sparse_input/NotEqualNotEqual]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_2Wlinear/linear_model/linear_model/linear_model/occupation/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:?????????
?
Plinear/linear_model/linear_model/linear_model/occupation/to_sparse_input/indicesWhereQlinear/linear_model/linear_model/linear_model/occupation/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Olinear/linear_model/linear_model/linear_model/occupation/to_sparse_input/valuesGatherNd]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_2Plinear/linear_model/linear_model/linear_model/occupation/to_sparse_input/indices*#
_output_shapes
:?????????*
Tindices0	*
Tparams0
?
Tlinear/linear_model/linear_model/linear_model/occupation/to_sparse_input/dense_shapeShape]linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/ExpandDims_2*
T0*
out_type0	*
_output_shapes
:
?
?linear/linear_model/linear_model/linear_model/occupation/lookupStringToHashBucketFastOlinear/linear_model/linear_model/linear_model/occupation/to_sparse_input/values*
num_bucketsd*#
_output_shapes
:?????????
?
Clinear/linear_model/linear_model/linear_model/occupation/Shape/CastCastTlinear/linear_model/linear_model/linear_model/occupation/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
?
Llinear/linear_model/linear_model/linear_model/occupation/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/occupation/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/occupation/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Flinear/linear_model/linear_model/linear_model/occupation/strided_sliceStridedSliceClinear/linear_model/linear_model/linear_model/occupation/Shape/CastLlinear/linear_model/linear_model/linear_model/occupation/strided_slice/stackNlinear/linear_model/linear_model/linear_model/occupation/strided_slice/stack_1Nlinear/linear_model/linear_model/linear_model/occupation/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
?
Alinear/linear_model/linear_model/linear_model/occupation/Cast/x/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
?linear/linear_model/linear_model/linear_model/occupation/Cast/xPackFlinear/linear_model/linear_model/linear_model/occupation/strided_sliceAlinear/linear_model/linear_model/linear_model/occupation/Cast/x/1*
T0*
N*
_output_shapes
:
?
=linear/linear_model/linear_model/linear_model/occupation/CastCast?linear/linear_model/linear_model/linear_model/occupation/Cast/x*

SrcT0*
_output_shapes
:*

DstT0	
?
Flinear/linear_model/linear_model/linear_model/occupation/SparseReshapeSparseReshapePlinear/linear_model/linear_model/linear_model/occupation/to_sparse_input/indicesTlinear/linear_model/linear_model/linear_model/occupation/to_sparse_input/dense_shape=linear/linear_model/linear_model/linear_model/occupation/Cast*-
_output_shapes
:?????????:
?
Olinear/linear_model/linear_model/linear_model/occupation/SparseReshape/IdentityIdentity?linear/linear_model/linear_model/linear_model/occupation/lookup*
T0	*#
_output_shapes
:?????????
?
Qlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
Plinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SliceSliceHlinear/linear_model/linear_model/linear_model/occupation/SparseReshape:1Qlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice/beginPlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ProdProdKlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SliceKlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Const*
T0	*
_output_shapes
: 
?
Vlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2/indicesConst*
dtype0*
_output_shapes
: *
value	B :
?
Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Nlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2GatherV2Hlinear/linear_model/linear_model/linear_model/occupation/SparseReshape:1Vlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2/indicesSlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Taxis0*
Tindices0
?
Llinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Cast/xPackJlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ProdNlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2*
N*
_output_shapes
:*
T0	
?
Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseReshapeSparseReshapeFlinear/linear_model/linear_model/linear_model/occupation/SparseReshapeHlinear/linear_model/linear_model/linear_model/occupation/SparseReshape:1Llinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
\linear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseReshape/IdentityIdentityOlinear/linear_model/linear_model/linear_model/occupation/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
Tlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
Rlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GreaterEqualGreaterEqual\linear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseReshape/IdentityTlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GreaterEqual/y*#
_output_shapes
:?????????*
T0	
?
Klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/WhereWhereRlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?
Mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ReshapeReshapeKlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/WhereSlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
Ulinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Plinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_1GatherV2Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseReshapeMlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ReshapeUlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_1/axis*'
_output_shapes
:?????????*
Taxis0*
Tindices0	*
Tparams0	
?
Ulinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Plinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_2GatherV2\linear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseReshape/IdentityMlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ReshapeUlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_2/axis*
Tindices0	*
Tparams0	*#
_output_shapes
:?????????*
Taxis0
?
Nlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/IdentityIdentityUlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
_linear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsPlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_1Plinear/linear_model/linear_model/linear_model/occupation/weighted_sum/GatherV2_2Nlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Identity_linear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseFillEmptyRows/Const*T
_output_shapesB
@:?????????:?????????:?????????:?????????*
T0	
?
qlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
?
slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
?
klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSlicemlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsqlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stackslinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
end_mask*#
_output_shapes
:?????????*
T0	*
Index0*
shrink_axis_mask*

begin_mask
?
blinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/CastCastklinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/strided_slice*#
_output_shapes
:?????????*

DstT0*

SrcT0	
?
dlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/UniqueUniqueolinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
nlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather-linear/linear_model/occupation/weights/part_0dlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0	*@
_class6
42loc:@linear/linear_model/occupation/weights/part_0*
dtype0*'
_output_shapes
:?????????
?
wlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentitynlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*@
_class6
42loc:@linear/linear_model/occupation/weights/part_0*'
_output_shapes
:?????????
?
ylinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identitywlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*'
_output_shapes
:?????????*
T0
?
]linear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparseSparseSegmentSumylinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1flinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/Unique:1blinear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse/Cast*'
_output_shapes
:?????????*
T0
?
Ulinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape_1/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape_1Reshapeolinear/linear_model/linear_model/linear_model/occupation/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Ulinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape_1/shape*
T0
*'
_output_shapes
:?????????
?
Klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ShapeShape]linear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
Ylinear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
[linear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
[linear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_sliceStridedSliceKlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/ShapeYlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_slice/stack[linear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_slice/stack_1[linear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_slice/stack_2*
T0*
Index0*
_output_shapes
: *
shrink_axis_mask
?
Mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
?
Klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/stackPackMlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/stack/0Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/strided_slice*
T0*
N*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/TileTileOlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape_1Klinear/linear_model/linear_model/linear_model/occupation/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
Plinear/linear_model/linear_model/linear_model/occupation/weighted_sum/zeros_like	ZerosLike]linear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse*'
_output_shapes
:?????????*
T0
?
Elinear/linear_model/linear_model/linear_model/occupation/weighted_sumSelectJlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/TilePlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/zeros_like]linear/linear_model/linear_model/linear_model/occupation/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Llinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Cast_1CastHlinear/linear_model/linear_model/linear_model/occupation/SparseReshape:1*
_output_shapes
:*

DstT0*

SrcT0	
?
Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_1SliceLlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Cast_1Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_1/beginRlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_1/size*
_output_shapes
:*
Index0*
T0
?
Mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Shape_1ShapeElinear/linear_model/linear_model/linear_model/occupation/weighted_sum*
T0*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_2/sizeConst*
dtype0*
_output_shapes
:*
valueB:
?????????
?
Mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_2SliceMlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Shape_1Slinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_2/beginRlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Llinear/linear_model/linear_model/linear_model/occupation/weighted_sum/concatConcatV2Mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_1Mlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Slice_2Qlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/concat/axis*
T0*
N*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape_2ReshapeElinear/linear_model/linear_model/linear_model/occupation/weighted_sumLlinear/linear_model/linear_model/linear_model/occupation/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
Ilinear/linear_model/linear_model/linear_model/relationship/ExpandDims/dimConst*
dtype0*
_output_shapes
: *
valueB :
?????????
?
Elinear/linear_model/linear_model/linear_model/relationship/ExpandDims
ExpandDimsPlaceholder_4Ilinear/linear_model/linear_model/linear_model/relationship/ExpandDims/dim*'
_output_shapes
:?????????*
T0
?
Ylinear/linear_model/linear_model/linear_model/relationship/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
Slinear/linear_model/linear_model/linear_model/relationship/to_sparse_input/NotEqualNotEqualElinear/linear_model/linear_model/linear_model/relationship/ExpandDimsYlinear/linear_model/linear_model/linear_model/relationship/to_sparse_input/ignore_value/x*'
_output_shapes
:?????????*
T0
?
Rlinear/linear_model/linear_model/linear_model/relationship/to_sparse_input/indicesWhereSlinear/linear_model/linear_model/linear_model/relationship/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Qlinear/linear_model/linear_model/linear_model/relationship/to_sparse_input/valuesGatherNdElinear/linear_model/linear_model/linear_model/relationship/ExpandDimsRlinear/linear_model/linear_model/linear_model/relationship/to_sparse_input/indices*
Tindices0	*
Tparams0*#
_output_shapes
:?????????
?
Vlinear/linear_model/linear_model/linear_model/relationship/to_sparse_input/dense_shapeShapeElinear/linear_model/linear_model/linear_model/relationship/ExpandDims*
T0*
out_type0	*
_output_shapes
:
?
Tlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/ConstConst*]
valueTBRB HusbandB Not-in-familyB WifeB
 Own-childB
 UnmarriedB Other-relative*
dtype0*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/SizeConst*
value	B :*
dtype0*
_output_shapes
: 
?
Zlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/range/startConst*
dtype0*
_output_shapes
: *
value	B : 
?
Zlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
Tlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/rangeRangeZlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/range/startSlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/SizeZlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/range/delta*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/CastCastTlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/range*

SrcT0*
_output_shapes
:*

DstT0	
?
_linear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/ConstConst*
dtype0	*
_output_shapes
: *
valueB	 R
?????????
?
dlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/hash_tableHashTableV2*
	key_dtype0*
value_dtype0	*
_output_shapes
: 
?
xlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2dlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/hash_tableTlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/ConstSlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/Cast*	
Tin0*

Tout0	
?
^linear/linear_model/linear_model/linear_model/relationship/hash_table_Lookup/LookupTableFindV2LookupTableFindV2dlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/hash_tableQlinear/linear_model/linear_model/linear_model/relationship/to_sparse_input/values_linear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/Const*#
_output_shapes
:?????????*	
Tin0*

Tout0	
?
Elinear/linear_model/linear_model/linear_model/relationship/Shape/CastCastVlinear/linear_model/linear_model/linear_model/relationship/to_sparse_input/dense_shape*
_output_shapes
:*

DstT0*

SrcT0	
?
Nlinear/linear_model/linear_model/linear_model/relationship/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Plinear/linear_model/linear_model/linear_model/relationship/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
Plinear/linear_model/linear_model/linear_model/relationship/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Hlinear/linear_model/linear_model/linear_model/relationship/strided_sliceStridedSliceElinear/linear_model/linear_model/linear_model/relationship/Shape/CastNlinear/linear_model/linear_model/linear_model/relationship/strided_slice/stackPlinear/linear_model/linear_model/linear_model/relationship/strided_slice/stack_1Plinear/linear_model/linear_model/linear_model/relationship/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
?
Clinear/linear_model/linear_model/linear_model/relationship/Cast/x/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Alinear/linear_model/linear_model/linear_model/relationship/Cast/xPackHlinear/linear_model/linear_model/linear_model/relationship/strided_sliceClinear/linear_model/linear_model/linear_model/relationship/Cast/x/1*
T0*
N*
_output_shapes
:
?
?linear/linear_model/linear_model/linear_model/relationship/CastCastAlinear/linear_model/linear_model/linear_model/relationship/Cast/x*

SrcT0*
_output_shapes
:*

DstT0	
?
Hlinear/linear_model/linear_model/linear_model/relationship/SparseReshapeSparseReshapeRlinear/linear_model/linear_model/linear_model/relationship/to_sparse_input/indicesVlinear/linear_model/linear_model/linear_model/relationship/to_sparse_input/dense_shape?linear/linear_model/linear_model/linear_model/relationship/Cast*-
_output_shapes
:?????????:
?
Qlinear/linear_model/linear_model/linear_model/relationship/SparseReshape/IdentityIdentity^linear/linear_model/linear_model/linear_model/relationship/hash_table_Lookup/LookupTableFindV2*#
_output_shapes
:?????????*
T0	
?
Slinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice/beginConst*
dtype0*
_output_shapes
:*
valueB: 
?
Rlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SliceSliceJlinear/linear_model/linear_model/linear_model/relationship/SparseReshape:1Slinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice/beginRlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice/size*
_output_shapes
:*
Index0*
T0	
?
Mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/ConstConst*
dtype0*
_output_shapes
:*
valueB: 
?
Llinear/linear_model/linear_model/linear_model/relationship/weighted_sum/ProdProdMlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SliceMlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Const*
T0	*
_output_shapes
: 
?
Xlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Plinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2GatherV2Jlinear/linear_model/linear_model/linear_model/relationship/SparseReshape:1Xlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2/indicesUlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2/axis*
Tparams0	*
_output_shapes
: *
Taxis0*
Tindices0
?
Nlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Cast/xPackLlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/ProdPlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2*
T0	*
N*
_output_shapes
:
?
Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseReshapeSparseReshapeHlinear/linear_model/linear_model/linear_model/relationship/SparseReshapeJlinear/linear_model/linear_model/linear_model/relationship/SparseReshape:1Nlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
^linear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseReshape/IdentityIdentityQlinear/linear_model/linear_model/linear_model/relationship/SparseReshape/Identity*#
_output_shapes
:?????????*
T0	
?
Vlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
Tlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GreaterEqualGreaterEqual^linear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseReshape/IdentityVlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
Mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/WhereWhereTlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/ReshapeReshapeMlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/WhereUlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape/shape*
T0	*#
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Rlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_1GatherV2Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseReshapeOlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/ReshapeWlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_1/axis*
Tparams0	*'
_output_shapes
:?????????*
Taxis0*
Tindices0	
?
Wlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Rlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_2GatherV2^linear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseReshape/IdentityOlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/ReshapeWlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_2/axis*
Tparams0	*#
_output_shapes
:?????????*
Taxis0*
Tindices0	
?
Plinear/linear_model/linear_model/linear_model/relationship/weighted_sum/IdentityIdentityWlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseReshape:1*
_output_shapes
:*
T0	
?
alinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsRlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_1Rlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/GatherV2_2Plinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Identityalinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
slinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
?
ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
valueB"      *
dtype0*
_output_shapes
:
?
mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSliceolinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsslinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_slice/stackulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:?????????*
Index0*
T0	
?
dlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/CastCastmlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:?????????*

DstT0
?
flinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/UniqueUniqueqlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*2
_output_shapes 
:?????????:?????????*
T0	
?
plinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather/linear/linear_model/relationship/weights/part_0flinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/Unique*B
_class8
64loc:@linear/linear_model/relationship/weights/part_0*
dtype0*'
_output_shapes
:?????????*
Tindices0	
?
ylinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentityplinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*B
_class8
64loc:@linear/linear_model/relationship/weights/part_0*'
_output_shapes
:?????????
?
{linear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identityylinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*'
_output_shapes
:?????????*
T0
?
_linear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparseSparseSegmentSum{linear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1hlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/Unique:1dlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:?????????
?
Wlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape_1/shapeConst*
dtype0*
_output_shapes
:*
valueB"????   
?
Qlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape_1Reshapeqlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Wlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape_1/shape*'
_output_shapes
:?????????*
T0

?
Mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/ShapeShape_linear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse*
_output_shapes
:*
T0
?
[linear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
]linear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
?
]linear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_sliceStridedSliceMlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Shape[linear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_slice/stack]linear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_slice/stack_1]linear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
?
Olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/stack/0Const*
dtype0*
_output_shapes
: *
value	B :
?
Mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/stackPackOlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/stack/0Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/strided_slice*
T0*
N*
_output_shapes
:
?
Llinear/linear_model/linear_model/linear_model/relationship/weighted_sum/TileTileQlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape_1Mlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/stack*0
_output_shapes
:??????????????????*
T0

?
Rlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/zeros_like	ZerosLike_linear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Glinear/linear_model/linear_model/linear_model/relationship/weighted_sumSelectLlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/TileRlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/zeros_like_linear/linear_model/linear_model/linear_model/relationship/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Nlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Cast_1CastJlinear/linear_model/linear_model/linear_model/relationship/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
?
Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_1/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
Tlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_1SliceNlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Cast_1Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_1/beginTlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_1/size*
Index0*
T0*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Shape_1ShapeGlinear/linear_model/linear_model/linear_model/relationship/weighted_sum*
T0*
_output_shapes
:
?
Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
?
Tlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_2/sizeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_2SliceOlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Shape_1Ulinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_2/beginTlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
Slinear/linear_model/linear_model/linear_model/relationship/weighted_sum/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Nlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/concatConcatV2Olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_1Olinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Slice_2Slinear/linear_model/linear_model/linear_model/relationship/weighted_sum/concat/axis*
T0*
N*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape_2ReshapeGlinear/linear_model/linear_model/linear_model/relationship/weighted_sumNlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/concat*
T0*'
_output_shapes
:?????????
?
Flinear/linear_model/linear_model/linear_model/workclass/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
Blinear/linear_model/linear_model/linear_model/workclass/ExpandDims
ExpandDimsPlaceholder_5Flinear/linear_model/linear_model/linear_model/workclass/ExpandDims/dim*
T0*'
_output_shapes
:?????????
?
Vlinear/linear_model/linear_model/linear_model/workclass/to_sparse_input/ignore_value/xConst*
valueB B *
dtype0*
_output_shapes
: 
?
Plinear/linear_model/linear_model/linear_model/workclass/to_sparse_input/NotEqualNotEqualBlinear/linear_model/linear_model/linear_model/workclass/ExpandDimsVlinear/linear_model/linear_model/linear_model/workclass/to_sparse_input/ignore_value/x*
T0*'
_output_shapes
:?????????
?
Olinear/linear_model/linear_model/linear_model/workclass/to_sparse_input/indicesWherePlinear/linear_model/linear_model/linear_model/workclass/to_sparse_input/NotEqual*'
_output_shapes
:?????????
?
Nlinear/linear_model/linear_model/linear_model/workclass/to_sparse_input/valuesGatherNdBlinear/linear_model/linear_model/linear_model/workclass/ExpandDimsOlinear/linear_model/linear_model/linear_model/workclass/to_sparse_input/indices*
Tparams0*#
_output_shapes
:?????????*
Tindices0	
?
Slinear/linear_model/linear_model/linear_model/workclass/to_sparse_input/dense_shapeShapeBlinear/linear_model/linear_model/linear_model/workclass/ExpandDims*
T0*
out_type0	*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/ConstConst*
dtype0*
_output_shapes
:	*?
value}B{	B Self-emp-not-incB PrivateB
 State-govB Federal-govB
 Local-govB ?B Self-emp-incB Without-payB Never-worked
?
Mlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/SizeConst*
value	B :	*
dtype0*
_output_shapes
: 
?
Tlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
?
Tlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
Nlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/rangeRangeTlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/range/startMlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/SizeTlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/range/delta*
_output_shapes
:	
?
Mlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/CastCastNlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/range*

SrcT0*
_output_shapes
:	*

DstT0	
?
Ylinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/hash_table/ConstConst*
dtype0	*
_output_shapes
: *
valueB	 R
?????????
?
^linear/linear_model/linear_model/linear_model/workclass/workclass_lookup/hash_table/hash_tableHashTableV2*
	key_dtype0*
value_dtype0	*
_output_shapes
: 
?
rlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/hash_table/table_init/LookupTableImportV2LookupTableImportV2^linear/linear_model/linear_model/linear_model/workclass/workclass_lookup/hash_table/hash_tableNlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/ConstMlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/Cast*

Tout0	*	
Tin0
?
[linear/linear_model/linear_model/linear_model/workclass/hash_table_Lookup/LookupTableFindV2LookupTableFindV2^linear/linear_model/linear_model/linear_model/workclass/workclass_lookup/hash_table/hash_tableNlinear/linear_model/linear_model/linear_model/workclass/to_sparse_input/valuesYlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/hash_table/Const*

Tout0	*#
_output_shapes
:?????????*	
Tin0
?
Blinear/linear_model/linear_model/linear_model/workclass/Shape/CastCastSlinear/linear_model/linear_model/linear_model/workclass/to_sparse_input/dense_shape*

SrcT0	*
_output_shapes
:*

DstT0
?
Klinear/linear_model/linear_model/linear_model/workclass/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
?
Mlinear/linear_model/linear_model/linear_model/workclass/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
?
Mlinear/linear_model/linear_model/linear_model/workclass/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
Elinear/linear_model/linear_model/linear_model/workclass/strided_sliceStridedSliceBlinear/linear_model/linear_model/linear_model/workclass/Shape/CastKlinear/linear_model/linear_model/linear_model/workclass/strided_slice/stackMlinear/linear_model/linear_model/linear_model/workclass/strided_slice/stack_1Mlinear/linear_model/linear_model/linear_model/workclass/strided_slice/stack_2*
_output_shapes
: *
shrink_axis_mask*
Index0*
T0
?
@linear/linear_model/linear_model/linear_model/workclass/Cast/x/1Const*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
>linear/linear_model/linear_model/linear_model/workclass/Cast/xPackElinear/linear_model/linear_model/linear_model/workclass/strided_slice@linear/linear_model/linear_model/linear_model/workclass/Cast/x/1*
T0*
N*
_output_shapes
:
?
<linear/linear_model/linear_model/linear_model/workclass/CastCast>linear/linear_model/linear_model/linear_model/workclass/Cast/x*

SrcT0*
_output_shapes
:*

DstT0	
?
Elinear/linear_model/linear_model/linear_model/workclass/SparseReshapeSparseReshapeOlinear/linear_model/linear_model/linear_model/workclass/to_sparse_input/indicesSlinear/linear_model/linear_model/linear_model/workclass/to_sparse_input/dense_shape<linear/linear_model/linear_model/linear_model/workclass/Cast*-
_output_shapes
:?????????:
?
Nlinear/linear_model/linear_model/linear_model/workclass/SparseReshape/IdentityIdentity[linear/linear_model/linear_model/linear_model/workclass/hash_table_Lookup/LookupTableFindV2*
T0	*#
_output_shapes
:?????????
?
Plinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice/beginConst*
valueB: *
dtype0*
_output_shapes
:
?
Olinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SliceSliceGlinear/linear_model/linear_model/linear_model/workclass/SparseReshape:1Plinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice/beginOlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice/size*
Index0*
T0	*
_output_shapes
:
?
Jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ConstConst*
valueB: *
dtype0*
_output_shapes
:
?
Ilinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ProdProdJlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SliceJlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Const*
_output_shapes
: *
T0	
?
Ulinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2/indicesConst*
value	B :*
dtype0*
_output_shapes
: 
?
Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
Mlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2GatherV2Glinear/linear_model/linear_model/linear_model/workclass/SparseReshape:1Ulinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2/indicesRlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2/axis*
Taxis0*
Tindices0*
Tparams0	*
_output_shapes
: 
?
Klinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Cast/xPackIlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ProdMlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2*
N*
_output_shapes
:*
T0	
?
Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseReshapeSparseReshapeElinear/linear_model/linear_model/linear_model/workclass/SparseReshapeGlinear/linear_model/linear_model/linear_model/workclass/SparseReshape:1Klinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Cast/x*-
_output_shapes
:?????????:
?
[linear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseReshape/IdentityIdentityNlinear/linear_model/linear_model/linear_model/workclass/SparseReshape/Identity*
T0	*#
_output_shapes
:?????????
?
Slinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GreaterEqual/yConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
Qlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GreaterEqualGreaterEqual[linear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseReshape/IdentitySlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GreaterEqual/y*
T0	*#
_output_shapes
:?????????
?
Jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/WhereWhereQlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GreaterEqual*'
_output_shapes
:?????????
?
Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape/shapeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ReshapeReshapeJlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/WhereRlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape/shape*#
_output_shapes
:?????????*
T0	
?
Tlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_1/axisConst*
dtype0*
_output_shapes
: *
value	B : 
?
Olinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_1GatherV2Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseReshapeLlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ReshapeTlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_1/axis*
Tparams0	*'
_output_shapes
:?????????*
Taxis0*
Tindices0	
?
Tlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_2/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Olinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_2GatherV2[linear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseReshape/IdentityLlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ReshapeTlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_2/axis*
Tparams0	*#
_output_shapes
:?????????*
Taxis0*
Tindices0	
?
Mlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/IdentityIdentityTlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseReshape:1*
T0	*
_output_shapes
:
?
^linear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseFillEmptyRows/ConstConst*
value	B	 R *
dtype0	*
_output_shapes
: 
?
llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsSparseFillEmptyRowsOlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_1Olinear/linear_model/linear_model/linear_model/workclass/weighted_sum/GatherV2_2Mlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Identity^linear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseFillEmptyRows/Const*
T0	*T
_output_shapesB
@:?????????:?????????:?????????:?????????
?
plinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_slice/stackConst*
valueB"        *
dtype0*
_output_shapes
:
?
rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1Const*
valueB"       *
dtype0*
_output_shapes
:
?
rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB"      
?
jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_sliceStridedSlicellinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRowsplinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_slice/stackrlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_slice/stack_1rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_slice/stack_2*
shrink_axis_mask*

begin_mask*
end_mask*#
_output_shapes
:?????????*
T0	*
Index0
?
alinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/CastCastjlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/strided_slice*

SrcT0	*#
_output_shapes
:?????????*

DstT0
?
clinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/UniqueUniquenlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:1*
T0	*2
_output_shapes 
:?????????:?????????
?
mlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/embedding_lookupResourceGather,linear/linear_model/workclass/weights/part_0clinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/Unique*
Tindices0	*?
_class5
31loc:@linear/linear_model/workclass/weights/part_0*
dtype0*'
_output_shapes
:?????????
?
vlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/embedding_lookup/IdentityIdentitymlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/embedding_lookup*
T0*?
_class5
31loc:@linear/linear_model/workclass/weights/part_0*'
_output_shapes
:?????????
?
xlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1Identityvlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity*
T0*'
_output_shapes
:?????????
?
\linear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparseSparseSegmentSumxlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/embedding_lookup/Identity_1elinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/Unique:1alinear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse/Cast*
T0*'
_output_shapes
:?????????
?
Tlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape_1/shapeConst*
valueB"????   *
dtype0*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape_1Reshapenlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/SparseFillEmptyRows/SparseFillEmptyRows:2Tlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape_1/shape*'
_output_shapes
:?????????*
T0

?
Jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ShapeShape\linear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse*
T0*
_output_shapes
:
?
Xlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
?
Zlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
?
Zlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_sliceStridedSliceJlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/ShapeXlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_slice/stackZlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_slice/stack_1Zlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
?
Llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/stack/0Const*
value	B :*
dtype0*
_output_shapes
: 
?
Jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/stackPackLlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/stack/0Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/strided_slice*
N*
_output_shapes
:*
T0
?
Ilinear/linear_model/linear_model/linear_model/workclass/weighted_sum/TileTileNlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape_1Jlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/stack*
T0
*0
_output_shapes
:??????????????????
?
Olinear/linear_model/linear_model/linear_model/workclass/weighted_sum/zeros_like	ZerosLike\linear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Dlinear/linear_model/linear_model/linear_model/workclass/weighted_sumSelectIlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/TileOlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/zeros_like\linear/linear_model/linear_model/linear_model/workclass/weighted_sum/embedding_lookup_sparse*
T0*'
_output_shapes
:?????????
?
Klinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Cast_1CastGlinear/linear_model/linear_model/linear_model/workclass/SparseReshape:1*

SrcT0	*
_output_shapes
:*

DstT0
?
Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_1/beginConst*
dtype0*
_output_shapes
:*
valueB: 
?
Qlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_1/sizeConst*
valueB:*
dtype0*
_output_shapes
:
?
Llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_1SliceKlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Cast_1Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_1/beginQlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_1/size*
_output_shapes
:*
Index0*
T0
?
Llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Shape_1ShapeDlinear/linear_model/linear_model/linear_model/workclass/weighted_sum*
T0*
_output_shapes
:
?
Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_2/beginConst*
valueB:*
dtype0*
_output_shapes
:
?
Qlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_2/sizeConst*
valueB:
?????????*
dtype0*
_output_shapes
:
?
Llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_2SliceLlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Shape_1Rlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_2/beginQlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_2/size*
Index0*
T0*
_output_shapes
:
?
Plinear/linear_model/linear_model/linear_model/workclass/weighted_sum/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
?
Klinear/linear_model/linear_model/linear_model/workclass/weighted_sum/concatConcatV2Llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_1Llinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Slice_2Plinear/linear_model/linear_model/linear_model/workclass/weighted_sum/concat/axis*
T0*
N*
_output_shapes
:
?
Nlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape_2ReshapeDlinear/linear_model/linear_model/linear_model/workclass/weighted_sumKlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/concat*'
_output_shapes
:?????????*
T0
?
Blinear/linear_model/linear_model/linear_model/weighted_sum_no_biasAddNSlinear/linear_model/linear_model/linear_model/age_bucketized/weighted_sum/Reshape_2glinear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/weighted_sum/Reshape_2Nlinear/linear_model/linear_model/linear_model/education/weighted_sum/Reshape_2[linear/linear_model/linear_model/linear_model/education_X_occupation/weighted_sum/Reshape_2Klinear/linear_model/linear_model/linear_model/gender/weighted_sum/Reshape_2Slinear/linear_model/linear_model/linear_model/marital_status/weighted_sum/Reshape_2Slinear/linear_model/linear_model/linear_model/native_country/weighted_sum/Reshape_2`linear/linear_model/linear_model/linear_model/native_country_X_occupation/weighted_sum/Reshape_2Olinear/linear_model/linear_model/linear_model/occupation/weighted_sum/Reshape_2Qlinear/linear_model/linear_model/linear_model/relationship/weighted_sum/Reshape_2Nlinear/linear_model/linear_model/linear_model/workclass/weighted_sum/Reshape_2*
T0*
N*'
_output_shapes
:?????????
?
/linear/linear_model/bias_weights/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:
?
 linear/linear_model/bias_weightsIdentity/linear/linear_model/bias_weights/ReadVariableOp*
T0*
_output_shapes
:
?
:linear/linear_model/linear_model/linear_model/weighted_sumBiasAddBlinear/linear_model/linear_model/linear_model/weighted_sum_no_bias linear/linear_model/bias_weights*
T0*'
_output_shapes
:?????????
y
linear/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:
d
linear/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
f
linear/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
f
linear/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
?
linear/strided_sliceStridedSlicelinear/ReadVariableOplinear/strided_slice/stacklinear/strided_slice/stack_1linear/strided_slice/stack_2*
shrink_axis_mask*
Index0*
T0*
_output_shapes
: 
\
linear/bias/tagsConst*
valueB Blinear/bias*
dtype0*
_output_shapes
: 
e
linear/biasScalarSummarylinear/bias/tagslinear/strided_slice*
_output_shapes
: *
T0
?
3linear/zero_fraction/total_size/Size/ReadVariableOpReadVariableOp1linear/linear_model/age_bucketized/weights/part_0*
dtype0*
_output_shapes

:
f
$linear/zero_fraction/total_size/SizeConst*
value	B	 R*
dtype0	*
_output_shapes
: 
?
5linear/zero_fraction/total_size/Size_1/ReadVariableOpReadVariableOpElinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0*
dtype0* 
_output_shapes
:
??=
j
&linear/zero_fraction/total_size/Size_1Const*
valueB		 R??=*
dtype0	*
_output_shapes
: 
?
5linear/zero_fraction/total_size/Size_2/ReadVariableOpReadVariableOp,linear/linear_model/education/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_2Const*
value	B	 R*
dtype0	*
_output_shapes
: 
?
5linear/zero_fraction/total_size/Size_3/ReadVariableOpReadVariableOp9linear/linear_model/education_X_occupation/weights/part_0*
dtype0*
_output_shapes
:	?N
i
&linear/zero_fraction/total_size/Size_3Const*
value
B	 R?N*
dtype0	*
_output_shapes
: 
?
5linear/zero_fraction/total_size/Size_4/ReadVariableOpReadVariableOp)linear/linear_model/gender/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_4Const*
value	B	 R*
dtype0	*
_output_shapes
: 
?
5linear/zero_fraction/total_size/Size_5/ReadVariableOpReadVariableOp1linear/linear_model/marital_status/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_5Const*
value	B	 R*
dtype0	*
_output_shapes
: 
?
5linear/zero_fraction/total_size/Size_6/ReadVariableOpReadVariableOp1linear/linear_model/native_country/weights/part_0*
dtype0*
_output_shapes

:d
h
&linear/zero_fraction/total_size/Size_6Const*
value	B	 Rd*
dtype0	*
_output_shapes
: 
?
5linear/zero_fraction/total_size/Size_7/ReadVariableOpReadVariableOp>linear/linear_model/native_country_X_occupation/weights/part_0*
dtype0*
_output_shapes
:	?N
i
&linear/zero_fraction/total_size/Size_7Const*
value
B	 R?N*
dtype0	*
_output_shapes
: 
?
5linear/zero_fraction/total_size/Size_8/ReadVariableOpReadVariableOp-linear/linear_model/occupation/weights/part_0*
dtype0*
_output_shapes

:d
h
&linear/zero_fraction/total_size/Size_8Const*
dtype0	*
_output_shapes
: *
value	B	 Rd
?
5linear/zero_fraction/total_size/Size_9/ReadVariableOpReadVariableOp/linear/linear_model/relationship/weights/part_0*
dtype0*
_output_shapes

:
h
&linear/zero_fraction/total_size/Size_9Const*
value	B	 R*
dtype0	*
_output_shapes
: 
?
6linear/zero_fraction/total_size/Size_10/ReadVariableOpReadVariableOp,linear/linear_model/workclass/weights/part_0*
dtype0*
_output_shapes

:	
i
'linear/zero_fraction/total_size/Size_10Const*
value	B	 R	*
dtype0	*
_output_shapes
: 
?
$linear/zero_fraction/total_size/AddNAddN$linear/zero_fraction/total_size/Size&linear/zero_fraction/total_size/Size_1&linear/zero_fraction/total_size/Size_2&linear/zero_fraction/total_size/Size_3&linear/zero_fraction/total_size/Size_4&linear/zero_fraction/total_size/Size_5&linear/zero_fraction/total_size/Size_6&linear/zero_fraction/total_size/Size_7&linear/zero_fraction/total_size/Size_8&linear/zero_fraction/total_size/Size_9'linear/zero_fraction/total_size/Size_10*
N*
_output_shapes
: *
T0	
g
%linear/zero_fraction/total_zero/ConstConst*
dtype0	*
_output_shapes
: *
value	B	 R 
?
%linear/zero_fraction/total_zero/EqualEqual$linear/zero_fraction/total_size/Size%linear/zero_fraction/total_zero/Const*
T0	*
_output_shapes
: 
?
1linear/zero_fraction/total_zero/zero_count/SwitchSwitch%linear/zero_fraction/total_zero/Equal%linear/zero_fraction/total_zero/Equal*
_output_shapes
: : *
T0

?
3linear/zero_fraction/total_zero/zero_count/switch_tIdentity3linear/zero_fraction/total_zero/zero_count/Switch:1*
T0
*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count/switch_fIdentity1linear/zero_fraction/total_zero/zero_count/Switch*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count/pred_idIdentity%linear/zero_fraction/total_zero/Equal*
T0
*
_output_shapes
: 
?
0linear/zero_fraction/total_zero/zero_count/ConstConst4^linear/zero_fraction/total_zero/zero_count/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpReadVariableOpNlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
?
Nlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/SwitchSwitch1linear/linear_model/age_bucketized/weights/part_02linear/zero_fraction/total_zero/zero_count/pred_id*
T0*D
_class:
86loc:@linear/linear_model/age_bucketized/weights/part_0*
_output_shapes
: : 
?
=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeConst4^linear/zero_fraction/total_zero/zero_count/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual/yConst4^linear/zero_fraction/total_zero/zero_count/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Blinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual	LessEqual=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeDlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/SwitchSwitchBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqualBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_tIdentityFlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

?
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_fIdentityDlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_idIdentityBlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual*
_output_shapes
: *
T0

?
Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zerosConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqualNotEqual]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchGlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpElinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*Z
_classP
NLloc:@linear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp
?
Plinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/CastCastTlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual*
_output_shapes

:*

DstT0*

SrcT0

?
Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/ConstConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
Ylinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_countSumPlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/CastQlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Blinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/CastCastYlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
?
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zerosConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchGlinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOpElinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*Z
_classP
NLloc:@linear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp
?
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/CastCastVlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual*
_output_shapes

:*

DstT0	*

SrcT0

?
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/ConstConstG^linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/CastSlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/MergeMerge[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_countBlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
Olinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/subSub=linear/zero_fraction/total_zero/zero_count/zero_fraction/SizeClinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Plinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/CastCastOlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
?
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast_1Cast=linear/zero_fraction/total_zero/zero_count/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
?
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/truedivRealDivPlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/CastRlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Alinear/zero_fraction/total_zero/zero_count/zero_fraction/fractionIdentitySlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count/ToFloatCast9linear/zero_fraction/total_zero/zero_count/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0
?
9linear/zero_fraction/total_zero/zero_count/ToFloat/SwitchSwitch$linear/zero_fraction/total_size/Size2linear/zero_fraction/total_zero/zero_count/pred_id*
_output_shapes
: : *
T0	*7
_class-
+)loc:@linear/zero_fraction/total_size/Size
?
.linear/zero_fraction/total_zero/zero_count/mulMulAlinear/zero_fraction/total_zero/zero_count/zero_fraction/fraction2linear/zero_fraction/total_zero/zero_count/ToFloat*
T0*
_output_shapes
: 
?
0linear/zero_fraction/total_zero/zero_count/MergeMerge.linear/zero_fraction/total_zero/zero_count/mul0linear/zero_fraction/total_zero/zero_count/Const*
T0*
N*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_1Const*
value	B	 R *
dtype0	*
_output_shapes
: 
?
'linear/zero_fraction/total_zero/Equal_1Equal&linear/zero_fraction/total_size/Size_1'linear/zero_fraction/total_zero/Const_1*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_1/SwitchSwitch'linear/zero_fraction/total_zero/Equal_1'linear/zero_fraction/total_zero/Equal_1*
_output_shapes
: : *
T0

?
5linear/zero_fraction/total_zero/zero_count_1/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_1/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_1/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_1/Switch*
_output_shapes
: *
T0

?
4linear/zero_fraction/total_zero/zero_count_1/pred_idIdentity'linear/zero_fraction/total_zero/Equal_1*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_1/ConstConst6^linear/zero_fraction/total_zero/zero_count_1/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/Switch*
dtype0* 
_output_shapes
:
??=
?
Plinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/SwitchSwitchElinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_04linear/zero_fraction/total_zero/zero_count_1/pred_id*
T0*X
_classN
LJloc:@linear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_1/switch_f*
valueB		 R??=*
dtype0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_1/switch_f*
dtype0	*
_output_shapes
: *
valueB	 R????
?
Dlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

?
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zeros*
T0* 
_output_shapes
:
??=
?
]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp*,
_output_shapes
:
??=:
??=
?
Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
* 
_output_shapes
:
??=*

DstT0
?
Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
?
[linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Xlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zeros* 
_output_shapes
:
??=*
T0
?
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp*,
_output_shapes
:
??=:
??=
?
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
* 
_output_shapes
:
??=*

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Cast*
N*
_output_shapes
: : *
T0	
?
Qlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
?
Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
?
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_1/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
?
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_1/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
?
4linear/zero_fraction/total_zero/zero_count_1/ToFloatCast;linear/zero_fraction/total_zero/zero_count_1/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0
?
;linear/zero_fraction/total_zero/zero_count_1/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_14linear/zero_fraction/total_zero/zero_count_1/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_1*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_1/mulMulClinear/zero_fraction/total_zero/zero_count_1/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_1/ToFloat*
_output_shapes
: *
T0
?
2linear/zero_fraction/total_zero/zero_count_1/MergeMerge0linear/zero_fraction/total_zero/zero_count_1/mul2linear/zero_fraction/total_zero/zero_count_1/Const*
N*
_output_shapes
: : *
T0
i
'linear/zero_fraction/total_zero/Const_2Const*
value	B	 R *
dtype0	*
_output_shapes
: 
?
'linear/zero_fraction/total_zero/Equal_2Equal&linear/zero_fraction/total_size/Size_2'linear/zero_fraction/total_zero/Const_2*
_output_shapes
: *
T0	
?
3linear/zero_fraction/total_zero/zero_count_2/SwitchSwitch'linear/zero_fraction/total_zero/Equal_2'linear/zero_fraction/total_zero/Equal_2*
_output_shapes
: : *
T0

?
5linear/zero_fraction/total_zero/zero_count_2/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_2/Switch:1*
_output_shapes
: *
T0

?
5linear/zero_fraction/total_zero/zero_count_2/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_2/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_2/pred_idIdentity'linear/zero_fraction/total_zero/Equal_2*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_2/ConstConst6^linear/zero_fraction/total_zero/zero_count_2/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
?
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
?
Plinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/SwitchSwitch,linear/linear_model/education/weights/part_04linear/zero_fraction/total_zero/zero_count_2/pred_id*
_output_shapes
: : *
T0*?
_class5
31loc:@linear/linear_model/education/weights/part_0
?
?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_2/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_2/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Switch*
_output_shapes
: *
T0

?
Glinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual*
_output_shapes
: *
T0

?
Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
?
Vlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp
?
Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
?
Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
[linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
?
Dlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
?
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Xlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id*(
_output_shapes
::*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp
?
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	
?
Elinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
?
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_2/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
?
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
?
Clinear/zero_fraction/total_zero/zero_count_2/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_2/ToFloatCast;linear/zero_fraction/total_zero/zero_count_2/ToFloat/Switch*
_output_shapes
: *

DstT0*

SrcT0	
?
;linear/zero_fraction/total_zero/zero_count_2/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_24linear/zero_fraction/total_zero/zero_count_2/pred_id*
_output_shapes
: : *
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_2
?
0linear/zero_fraction/total_zero/zero_count_2/mulMulClinear/zero_fraction/total_zero/zero_count_2/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_2/ToFloat*
T0*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_2/MergeMerge0linear/zero_fraction/total_zero/zero_count_2/mul2linear/zero_fraction/total_zero/zero_count_2/Const*
T0*
N*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_3Const*
value	B	 R *
dtype0	*
_output_shapes
: 
?
'linear/zero_fraction/total_zero/Equal_3Equal&linear/zero_fraction/total_size/Size_3'linear/zero_fraction/total_zero/Const_3*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_3/SwitchSwitch'linear/zero_fraction/total_zero/Equal_3'linear/zero_fraction/total_zero/Equal_3*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_3/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_3/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_3/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_3/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_3/pred_idIdentity'linear/zero_fraction/total_zero/Equal_3*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_3/ConstConst6^linear/zero_fraction/total_zero/zero_count_3/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
?
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes
:	?N
?
Plinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/SwitchSwitch9linear/linear_model/education_X_occupation/weights/part_04linear/zero_fraction/total_zero/zero_count_3/pred_id*
T0*L
_classB
@>loc:@linear/linear_model/education_X_occupation/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_3/switch_f*
value
B	 R?N*
dtype0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_3/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual*
_output_shapes
: : *
T0

?
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
?
Vlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes
:	?N
?
]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id**
_output_shapes
:	?N:	?N*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp
?
Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes
:	?N*

DstT0
?
Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
[linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
?
Xlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes
:	?N
?
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp**
_output_shapes
:	?N:	?N
?
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes
:	?N*

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
?
Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
?
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_3/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
?
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
?
Clinear/zero_fraction/total_zero/zero_count_3/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_3/ToFloatCast;linear/zero_fraction/total_zero/zero_count_3/ToFloat/Switch*
_output_shapes
: *

DstT0*

SrcT0	
?
;linear/zero_fraction/total_zero/zero_count_3/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_34linear/zero_fraction/total_zero/zero_count_3/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_3*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_3/mulMulClinear/zero_fraction/total_zero/zero_count_3/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_3/ToFloat*
T0*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_3/MergeMerge0linear/zero_fraction/total_zero/zero_count_3/mul2linear/zero_fraction/total_zero/zero_count_3/Const*
T0*
N*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_4Const*
value	B	 R *
dtype0	*
_output_shapes
: 
?
'linear/zero_fraction/total_zero/Equal_4Equal&linear/zero_fraction/total_size/Size_4'linear/zero_fraction/total_zero/Const_4*
_output_shapes
: *
T0	
?
3linear/zero_fraction/total_zero/zero_count_4/SwitchSwitch'linear/zero_fraction/total_zero/Equal_4'linear/zero_fraction/total_zero/Equal_4*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_4/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_4/Switch:1*
_output_shapes
: *
T0

?
5linear/zero_fraction/total_zero/zero_count_4/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_4/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_4/pred_idIdentity'linear/zero_fraction/total_zero/Equal_4*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_4/ConstConst6^linear/zero_fraction/total_zero/zero_count_4/switch_t*
dtype0*
_output_shapes
: *
valueB
 *    
?
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
?
Plinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp/SwitchSwitch)linear/linear_model/gender/weights/part_04linear/zero_fraction/total_zero/zero_count_4/pred_id*
_output_shapes
: : *
T0*<
_class2
0.loc:@linear/linear_model/gender/weights/part_0
?
?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_4/switch_f*
dtype0	*
_output_shapes
: *
value	B	 R
?
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_4/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual*
_output_shapes

:*

DstT0*

SrcT0

?
Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t*
dtype0*
_output_shapes
:*
valueB"       
?
[linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f*
dtype0*
_output_shapes
: *
valueB
 *    
?
Xlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes

:*
T0
?
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Const*
_output_shapes
: *
T0	
?
Elinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Cast*
N*
_output_shapes
: : *
T0	
?
Qlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
?
Rlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
?
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_4/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	
?
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_4/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_4/ToFloatCast;linear/zero_fraction/total_zero/zero_count_4/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0
?
;linear/zero_fraction/total_zero/zero_count_4/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_44linear/zero_fraction/total_zero/zero_count_4/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_4*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_4/mulMulClinear/zero_fraction/total_zero/zero_count_4/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_4/ToFloat*
_output_shapes
: *
T0
?
2linear/zero_fraction/total_zero/zero_count_4/MergeMerge0linear/zero_fraction/total_zero/zero_count_4/mul2linear/zero_fraction/total_zero/zero_count_4/Const*
T0*
N*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_5Const*
dtype0	*
_output_shapes
: *
value	B	 R 
?
'linear/zero_fraction/total_zero/Equal_5Equal&linear/zero_fraction/total_size/Size_5'linear/zero_fraction/total_zero/Const_5*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_5/SwitchSwitch'linear/zero_fraction/total_zero/Equal_5'linear/zero_fraction/total_zero/Equal_5*
_output_shapes
: : *
T0

?
5linear/zero_fraction/total_zero/zero_count_5/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_5/Switch:1*
_output_shapes
: *
T0

?
5linear/zero_fraction/total_zero/zero_count_5/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_5/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_5/pred_idIdentity'linear/zero_fraction/total_zero/Equal_5*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_5/ConstConst6^linear/zero_fraction/total_zero/zero_count_5/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
?
Plinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/SwitchSwitch1linear/linear_model/marital_status/weights/part_04linear/zero_fraction/total_zero/zero_count_5/pred_id*
T0*D
_class:
86loc:@linear/linear_model/marital_status/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_5/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_5/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
?
Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
[linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Xlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
?
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_5/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
?
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_5/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_5/ToFloatCast;linear/zero_fraction/total_zero/zero_count_5/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0
?
;linear/zero_fraction/total_zero/zero_count_5/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_54linear/zero_fraction/total_zero/zero_count_5/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_5*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_5/mulMulClinear/zero_fraction/total_zero/zero_count_5/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_5/ToFloat*
_output_shapes
: *
T0
?
2linear/zero_fraction/total_zero/zero_count_5/MergeMerge0linear/zero_fraction/total_zero/zero_count_5/mul2linear/zero_fraction/total_zero/zero_count_5/Const*
T0*
N*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_6Const*
dtype0	*
_output_shapes
: *
value	B	 R 
?
'linear/zero_fraction/total_zero/Equal_6Equal&linear/zero_fraction/total_size/Size_6'linear/zero_fraction/total_zero/Const_6*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_6/SwitchSwitch'linear/zero_fraction/total_zero/Equal_6'linear/zero_fraction/total_zero/Equal_6*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_6/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_6/Switch:1*
_output_shapes
: *
T0

?
5linear/zero_fraction/total_zero/zero_count_6/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_6/Switch*
_output_shapes
: *
T0

?
4linear/zero_fraction/total_zero/zero_count_6/pred_idIdentity'linear/zero_fraction/total_zero/Equal_6*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_6/ConstConst6^linear/zero_fraction/total_zero/zero_count_6/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:d
?
Plinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp/SwitchSwitch1linear/linear_model/native_country/weights/part_04linear/zero_fraction/total_zero/zero_count_6/pred_id*
_output_shapes
: : *
T0*D
_class:
86loc:@linear/linear_model/native_country/weights/part_0
?
?linear/zero_fraction/total_zero/zero_count_6/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_6/switch_f*
value	B	 Rd*
dtype0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_6/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_6/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual*
_output_shapes
: *
T0

?
Slinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:d
?
]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp*(
_output_shapes
:d:d
?
Rlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:d*

DstT0
?
Slinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
[linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
?
Dlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Xlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:d
?
_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp*(
_output_shapes
:d:d
?
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:d*

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_6/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
?
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_6/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
?
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
?
Clinear/zero_fraction/total_zero/zero_count_6/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_6/ToFloatCast;linear/zero_fraction/total_zero/zero_count_6/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0
?
;linear/zero_fraction/total_zero/zero_count_6/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_64linear/zero_fraction/total_zero/zero_count_6/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_6*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_6/mulMulClinear/zero_fraction/total_zero/zero_count_6/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_6/ToFloat*
_output_shapes
: *
T0
?
2linear/zero_fraction/total_zero/zero_count_6/MergeMerge0linear/zero_fraction/total_zero/zero_count_6/mul2linear/zero_fraction/total_zero/zero_count_6/Const*
N*
_output_shapes
: : *
T0
i
'linear/zero_fraction/total_zero/Const_7Const*
value	B	 R *
dtype0	*
_output_shapes
: 
?
'linear/zero_fraction/total_zero/Equal_7Equal&linear/zero_fraction/total_size/Size_7'linear/zero_fraction/total_zero/Const_7*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_7/SwitchSwitch'linear/zero_fraction/total_zero/Equal_7'linear/zero_fraction/total_zero/Equal_7*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_7/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_7/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_7/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_7/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_7/pred_idIdentity'linear/zero_fraction/total_zero/Equal_7*
_output_shapes
: *
T0

?
2linear/zero_fraction/total_zero/zero_count_7/ConstConst6^linear/zero_fraction/total_zero/zero_count_7/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes
:	?N
?
Plinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp/SwitchSwitch>linear/linear_model/native_country_X_occupation/weights/part_04linear/zero_fraction/total_zero/zero_count_7/pred_id*
T0*Q
_classG
ECloc:@linear/linear_model/native_country_X_occupation/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_7/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_7/switch_f*
value
B	 R?N*
dtype0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_7/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_7/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes
:	?N
?
]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp**
_output_shapes
:	?N:	?N
?
Rlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes
:	?N*

DstT0
?
Slinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
[linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Xlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes
:	?N*
T0
?
_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id**
_output_shapes
:	?N:	?N*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp
?
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes
:	?N*

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_7/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
?
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_7/zero_fraction/Size*
_output_shapes
: *

DstT0*

SrcT0	
?
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/Cast_1*
_output_shapes
: *
T0
?
Clinear/zero_fraction/total_zero/zero_count_7/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
?
4linear/zero_fraction/total_zero/zero_count_7/ToFloatCast;linear/zero_fraction/total_zero/zero_count_7/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0
?
;linear/zero_fraction/total_zero/zero_count_7/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_74linear/zero_fraction/total_zero/zero_count_7/pred_id*
_output_shapes
: : *
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_7
?
0linear/zero_fraction/total_zero/zero_count_7/mulMulClinear/zero_fraction/total_zero/zero_count_7/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_7/ToFloat*
_output_shapes
: *
T0
?
2linear/zero_fraction/total_zero/zero_count_7/MergeMerge0linear/zero_fraction/total_zero/zero_count_7/mul2linear/zero_fraction/total_zero/zero_count_7/Const*
T0*
N*
_output_shapes
: : 
i
'linear/zero_fraction/total_zero/Const_8Const*
dtype0	*
_output_shapes
: *
value	B	 R 
?
'linear/zero_fraction/total_zero/Equal_8Equal&linear/zero_fraction/total_size/Size_8'linear/zero_fraction/total_zero/Const_8*
_output_shapes
: *
T0	
?
3linear/zero_fraction/total_zero/zero_count_8/SwitchSwitch'linear/zero_fraction/total_zero/Equal_8'linear/zero_fraction/total_zero/Equal_8*
_output_shapes
: : *
T0

?
5linear/zero_fraction/total_zero/zero_count_8/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_8/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_8/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_8/Switch*
_output_shapes
: *
T0

?
4linear/zero_fraction/total_zero/zero_count_8/pred_idIdentity'linear/zero_fraction/total_zero/Equal_8*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_8/ConstConst6^linear/zero_fraction/total_zero/zero_count_8/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:d
?
Plinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp/SwitchSwitch-linear/linear_model/occupation/weights/part_04linear/zero_fraction/total_zero/zero_count_8/pred_id*
T0*@
_class6
42loc:@linear/linear_model/occupation/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_8/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_8/switch_f*
value	B	 Rd*
dtype0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_8/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_8/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqual/y*
_output_shapes
: *
T0	
?
Flinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Switch:1*
_output_shapes
: *
T0

?
Hlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/zeros*
_output_shapes

:d*
T0
?
]linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id*(
_output_shapes
:d:d*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp
?
Rlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:d*

DstT0
?
Slinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
[linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
?
Ulinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Xlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/zeros*
_output_shapes

:d*
T0
?
_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id*(
_output_shapes
:d:d*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp
?
Tlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:d*

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
]linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
Qlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_8/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/sub*
_output_shapes
: *

DstT0*

SrcT0	
?
Tlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_8/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
?
Ulinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_8/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/truediv*
_output_shapes
: *
T0
?
4linear/zero_fraction/total_zero/zero_count_8/ToFloatCast;linear/zero_fraction/total_zero/zero_count_8/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0
?
;linear/zero_fraction/total_zero/zero_count_8/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_84linear/zero_fraction/total_zero/zero_count_8/pred_id*
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_8*
_output_shapes
: : 
?
0linear/zero_fraction/total_zero/zero_count_8/mulMulClinear/zero_fraction/total_zero/zero_count_8/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_8/ToFloat*
T0*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_8/MergeMerge0linear/zero_fraction/total_zero/zero_count_8/mul2linear/zero_fraction/total_zero/zero_count_8/Const*
N*
_output_shapes
: : *
T0
i
'linear/zero_fraction/total_zero/Const_9Const*
value	B	 R *
dtype0	*
_output_shapes
: 
?
'linear/zero_fraction/total_zero/Equal_9Equal&linear/zero_fraction/total_size/Size_9'linear/zero_fraction/total_zero/Const_9*
T0	*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_9/SwitchSwitch'linear/zero_fraction/total_zero/Equal_9'linear/zero_fraction/total_zero/Equal_9*
T0
*
_output_shapes
: : 
?
5linear/zero_fraction/total_zero/zero_count_9/switch_tIdentity5linear/zero_fraction/total_zero/zero_count_9/Switch:1*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_9/switch_fIdentity3linear/zero_fraction/total_zero/zero_count_9/Switch*
T0
*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_9/pred_idIdentity'linear/zero_fraction/total_zero/Equal_9*
T0
*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_9/ConstConst6^linear/zero_fraction/total_zero/zero_count_9/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/ReadVariableOpReadVariableOpPlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:
?
Plinear/zero_fraction/total_zero/zero_count_9/zero_fraction/ReadVariableOp/SwitchSwitch/linear/linear_model/relationship/weights/part_04linear/zero_fraction/total_zero/zero_count_9/pred_id*
T0*B
_class8
64loc:@linear/linear_model/relationship/weights/part_0*
_output_shapes
: : 
?
?linear/zero_fraction/total_zero/zero_count_9/zero_fraction/SizeConst6^linear/zero_fraction/total_zero/zero_count_9/switch_f*
value	B	 R*
dtype0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqual/yConst6^linear/zero_fraction/total_zero/zero_count_9/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqual	LessEqual?linear/zero_fraction/total_zero/zero_count_9/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/SwitchSwitchDlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqualDlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqual*
T0
*
_output_shapes
: : 
?
Hlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_tIdentityHlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_fIdentityFlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Switch*
_output_shapes
: *
T0

?
Glinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_idIdentityDlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Slinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/zerosConstI^linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Vlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Slinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:
?
]linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_9/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Rlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/CastCastVlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0
?
Slinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/ConstConstI^linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
[linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/nonzero_countSumRlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/CastSlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/Const*
_output_shapes
: *
T0
?
Dlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/CastCast[linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/zerosConstI^linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Xlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchUlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:
?
_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchIlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/ReadVariableOpGlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id*
T0*\
_classR
PNloc:@linear/zero_fraction/total_zero/zero_count_9/zero_fraction/ReadVariableOp*(
_output_shapes
::
?
Tlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/CastCastXlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:*

DstT0	
?
Ulinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/ConstConstI^linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
]linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/nonzero_countSumTlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/CastUlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/MergeMerge]linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/nonzero_countDlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Cast*
N*
_output_shapes
: : *
T0	
?
Qlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/subSub?linear/zero_fraction/total_zero/zero_count_9/zero_fraction/SizeElinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Merge*
T0	*
_output_shapes
: 
?
Rlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/CastCastQlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
?
Tlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/Cast_1Cast?linear/zero_fraction/total_zero/zero_count_9/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
?
Ulinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/truedivRealDivRlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/CastTlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Clinear/zero_fraction/total_zero/zero_count_9/zero_fraction/fractionIdentityUlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_9/ToFloatCast;linear/zero_fraction/total_zero/zero_count_9/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0
?
;linear/zero_fraction/total_zero/zero_count_9/ToFloat/SwitchSwitch&linear/zero_fraction/total_size/Size_94linear/zero_fraction/total_zero/zero_count_9/pred_id*
_output_shapes
: : *
T0	*9
_class/
-+loc:@linear/zero_fraction/total_size/Size_9
?
0linear/zero_fraction/total_zero/zero_count_9/mulMulClinear/zero_fraction/total_zero/zero_count_9/zero_fraction/fraction4linear/zero_fraction/total_zero/zero_count_9/ToFloat*
T0*
_output_shapes
: 
?
2linear/zero_fraction/total_zero/zero_count_9/MergeMerge0linear/zero_fraction/total_zero/zero_count_9/mul2linear/zero_fraction/total_zero/zero_count_9/Const*
T0*
N*
_output_shapes
: : 
j
(linear/zero_fraction/total_zero/Const_10Const*
value	B	 R *
dtype0	*
_output_shapes
: 
?
(linear/zero_fraction/total_zero/Equal_10Equal'linear/zero_fraction/total_size/Size_10(linear/zero_fraction/total_zero/Const_10*
T0	*
_output_shapes
: 
?
4linear/zero_fraction/total_zero/zero_count_10/SwitchSwitch(linear/zero_fraction/total_zero/Equal_10(linear/zero_fraction/total_zero/Equal_10*
_output_shapes
: : *
T0

?
6linear/zero_fraction/total_zero/zero_count_10/switch_tIdentity6linear/zero_fraction/total_zero/zero_count_10/Switch:1*
T0
*
_output_shapes
: 
?
6linear/zero_fraction/total_zero/zero_count_10/switch_fIdentity4linear/zero_fraction/total_zero/zero_count_10/Switch*
T0
*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_10/pred_idIdentity(linear/zero_fraction/total_zero/Equal_10*
T0
*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_10/ConstConst7^linear/zero_fraction/total_zero/zero_count_10/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOpReadVariableOpQlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp/Switch*
dtype0*
_output_shapes

:	
?
Qlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp/SwitchSwitch,linear/linear_model/workclass/weights/part_05linear/zero_fraction/total_zero/zero_count_10/pred_id*
T0*?
_class5
31loc:@linear/linear_model/workclass/weights/part_0*
_output_shapes
: : 
?
@linear/zero_fraction/total_zero/zero_count_10/zero_fraction/SizeConst7^linear/zero_fraction/total_zero/zero_count_10/switch_f*
value	B	 R	*
dtype0	*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqual/yConst7^linear/zero_fraction/total_zero/zero_count_10/switch_f*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqual	LessEqual@linear/zero_fraction/total_zero/zero_count_10/zero_fraction/SizeGlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqual/y*
T0	*
_output_shapes
: 
?
Glinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/SwitchSwitchElinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqualElinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqual*
_output_shapes
: : *
T0

?
Ilinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_tIdentityIlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Switch:1*
T0
*
_output_shapes
: 
?
Ilinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_fIdentityGlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Switch*
T0
*
_output_shapes
: 
?
Hlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_idIdentityElinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqual*
T0
*
_output_shapes
: 
?
Tlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/zerosConstJ^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Wlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual/Switch:1Tlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/zeros*
T0*
_output_shapes

:	
?
^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp*(
_output_shapes
:	:	
?
Slinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/CastCastWlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual*
_output_shapes

:	*

DstT0*

SrcT0

?
Tlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/ConstConstJ^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
\linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/nonzero_countSumSlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/CastTlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
Elinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/CastCast\linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/nonzero_count*

SrcT0*
_output_shapes
: *

DstT0	
?
Vlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/zerosConstJ^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
Ylinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqualNotEqual`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchVlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/zeros*
T0*
_output_shapes

:	
?
`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual/SwitchSwitchJlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOpHlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id*
T0*]
_classS
QOloc:@linear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp*(
_output_shapes
:	:	
?
Ulinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/CastCastYlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual*

SrcT0
*
_output_shapes

:	*

DstT0	
?
Vlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/ConstConstJ^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/nonzero_countSumUlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/CastVlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
Flinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/MergeMerge^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/nonzero_countElinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
Rlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/subSub@linear/zero_fraction/total_zero/zero_count_10/zero_fraction/SizeFlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Merge*
_output_shapes
: *
T0	
?
Slinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/CastCastRlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
?
Ulinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/Cast_1Cast@linear/zero_fraction/total_zero/zero_count_10/zero_fraction/Size*

SrcT0	*
_output_shapes
: *

DstT0
?
Vlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/truedivRealDivSlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/CastUlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 
?
Dlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/fractionIdentityVlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
5linear/zero_fraction/total_zero/zero_count_10/ToFloatCast<linear/zero_fraction/total_zero/zero_count_10/ToFloat/Switch*

SrcT0	*
_output_shapes
: *

DstT0
?
<linear/zero_fraction/total_zero/zero_count_10/ToFloat/SwitchSwitch'linear/zero_fraction/total_size/Size_105linear/zero_fraction/total_zero/zero_count_10/pred_id*
T0	*:
_class0
.,loc:@linear/zero_fraction/total_size/Size_10*
_output_shapes
: : 
?
1linear/zero_fraction/total_zero/zero_count_10/mulMulDlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/fraction5linear/zero_fraction/total_zero/zero_count_10/ToFloat*
T0*
_output_shapes
: 
?
3linear/zero_fraction/total_zero/zero_count_10/MergeMerge1linear/zero_fraction/total_zero/zero_count_10/mul3linear/zero_fraction/total_zero/zero_count_10/Const*
T0*
N*
_output_shapes
: : 
?
$linear/zero_fraction/total_zero/AddNAddN0linear/zero_fraction/total_zero/zero_count/Merge2linear/zero_fraction/total_zero/zero_count_1/Merge2linear/zero_fraction/total_zero/zero_count_2/Merge2linear/zero_fraction/total_zero/zero_count_3/Merge2linear/zero_fraction/total_zero/zero_count_4/Merge2linear/zero_fraction/total_zero/zero_count_5/Merge2linear/zero_fraction/total_zero/zero_count_6/Merge2linear/zero_fraction/total_zero/zero_count_7/Merge2linear/zero_fraction/total_zero/zero_count_8/Merge2linear/zero_fraction/total_zero/zero_count_9/Merge3linear/zero_fraction/total_zero/zero_count_10/Merge*
T0*
N*
_output_shapes
: 
?
)linear/zero_fraction/compute/float32_sizeCast$linear/zero_fraction/total_size/AddN*
_output_shapes
: *

DstT0*

SrcT0	
?
$linear/zero_fraction/compute/truedivRealDiv$linear/zero_fraction/total_zero/AddN)linear/zero_fraction/compute/float32_size*
T0*
_output_shapes
: 
|
)linear/zero_fraction/zero_fraction_or_nanIdentity$linear/zero_fraction/compute/truediv*
T0*
_output_shapes
: 
?
$linear/fraction_of_zero_weights/tagsConst*0
value'B% Blinear/fraction_of_zero_weights*
dtype0*
_output_shapes
: 
?
linear/fraction_of_zero_weightsScalarSummary$linear/fraction_of_zero_weights/tags)linear/zero_fraction/zero_fraction_or_nan*
T0*
_output_shapes
: 
?
linear/zero_fraction_1/SizeSize:linear/linear_model/linear_model/linear_model/weighted_sum*
_output_shapes
: *
T0*
out_type0	
h
"linear/zero_fraction_1/LessEqual/yConst*
valueB	 R????*
dtype0	*
_output_shapes
: 
?
 linear/zero_fraction_1/LessEqual	LessEquallinear/zero_fraction_1/Size"linear/zero_fraction_1/LessEqual/y*
T0	*
_output_shapes
: 
?
"linear/zero_fraction_1/cond/SwitchSwitch linear/zero_fraction_1/LessEqual linear/zero_fraction_1/LessEqual*
_output_shapes
: : *
T0

w
$linear/zero_fraction_1/cond/switch_tIdentity$linear/zero_fraction_1/cond/Switch:1*
T0
*
_output_shapes
: 
u
$linear/zero_fraction_1/cond/switch_fIdentity"linear/zero_fraction_1/cond/Switch*
T0
*
_output_shapes
: 
r
#linear/zero_fraction_1/cond/pred_idIdentity linear/zero_fraction_1/LessEqual*
T0
*
_output_shapes
: 
?
/linear/zero_fraction_1/cond/count_nonzero/zerosConst%^linear/zero_fraction_1/cond/switch_t*
valueB
 *    *
dtype0*
_output_shapes
: 
?
2linear/zero_fraction_1/cond/count_nonzero/NotEqualNotEqual;linear/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1/linear/zero_fraction_1/cond/count_nonzero/zeros*'
_output_shapes
:?????????*
T0
?
9linear/zero_fraction_1/cond/count_nonzero/NotEqual/SwitchSwitch:linear/linear_model/linear_model/linear_model/weighted_sum#linear/zero_fraction_1/cond/pred_id*
T0*M
_classC
A?loc:@linear/linear_model/linear_model/linear_model/weighted_sum*:
_output_shapes(
&:?????????:?????????
?
.linear/zero_fraction_1/cond/count_nonzero/CastCast2linear/zero_fraction_1/cond/count_nonzero/NotEqual*

SrcT0
*'
_output_shapes
:?????????*

DstT0
?
/linear/zero_fraction_1/cond/count_nonzero/ConstConst%^linear/zero_fraction_1/cond/switch_t*
valueB"       *
dtype0*
_output_shapes
:
?
7linear/zero_fraction_1/cond/count_nonzero/nonzero_countSum.linear/zero_fraction_1/cond/count_nonzero/Cast/linear/zero_fraction_1/cond/count_nonzero/Const*
T0*
_output_shapes
: 
?
 linear/zero_fraction_1/cond/CastCast7linear/zero_fraction_1/cond/count_nonzero/nonzero_count*
_output_shapes
: *

DstT0	*

SrcT0
?
1linear/zero_fraction_1/cond/count_nonzero_1/zerosConst%^linear/zero_fraction_1/cond/switch_f*
valueB
 *    *
dtype0*
_output_shapes
: 
?
4linear/zero_fraction_1/cond/count_nonzero_1/NotEqualNotEqual;linear/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch1linear/zero_fraction_1/cond/count_nonzero_1/zeros*'
_output_shapes
:?????????*
T0
?
;linear/zero_fraction_1/cond/count_nonzero_1/NotEqual/SwitchSwitch:linear/linear_model/linear_model/linear_model/weighted_sum#linear/zero_fraction_1/cond/pred_id*
T0*M
_classC
A?loc:@linear/linear_model/linear_model/linear_model/weighted_sum*:
_output_shapes(
&:?????????:?????????
?
0linear/zero_fraction_1/cond/count_nonzero_1/CastCast4linear/zero_fraction_1/cond/count_nonzero_1/NotEqual*

SrcT0
*'
_output_shapes
:?????????*

DstT0	
?
1linear/zero_fraction_1/cond/count_nonzero_1/ConstConst%^linear/zero_fraction_1/cond/switch_f*
valueB"       *
dtype0*
_output_shapes
:
?
9linear/zero_fraction_1/cond/count_nonzero_1/nonzero_countSum0linear/zero_fraction_1/cond/count_nonzero_1/Cast1linear/zero_fraction_1/cond/count_nonzero_1/Const*
T0	*
_output_shapes
: 
?
!linear/zero_fraction_1/cond/MergeMerge9linear/zero_fraction_1/cond/count_nonzero_1/nonzero_count linear/zero_fraction_1/cond/Cast*
T0	*
N*
_output_shapes
: : 
?
-linear/zero_fraction_1/counts_to_fraction/subSublinear/zero_fraction_1/Size!linear/zero_fraction_1/cond/Merge*
_output_shapes
: *
T0	
?
.linear/zero_fraction_1/counts_to_fraction/CastCast-linear/zero_fraction_1/counts_to_fraction/sub*

SrcT0	*
_output_shapes
: *

DstT0
?
0linear/zero_fraction_1/counts_to_fraction/Cast_1Castlinear/zero_fraction_1/Size*

SrcT0	*
_output_shapes
: *

DstT0
?
1linear/zero_fraction_1/counts_to_fraction/truedivRealDiv.linear/zero_fraction_1/counts_to_fraction/Cast0linear/zero_fraction_1/counts_to_fraction/Cast_1*
T0*
_output_shapes
: 

linear/zero_fraction_1/fractionIdentity1linear/zero_fraction_1/counts_to_fraction/truediv*
T0*
_output_shapes
: 
?
*linear/linear/fraction_of_zero_values/tagsConst*6
value-B+ B%linear/linear/fraction_of_zero_values*
dtype0*
_output_shapes
: 
?
%linear/linear/fraction_of_zero_valuesScalarSummary*linear/linear/fraction_of_zero_values/tagslinear/zero_fraction_1/fraction*
T0*
_output_shapes
: 
u
linear/linear/activation/tagConst*)
value B Blinear/linear/activation*
dtype0*
_output_shapes
: 
?
linear/linear/activationHistogramSummarylinear/linear/activation/tag:linear/linear_model/linear_model/linear_model/weighted_sum*
_output_shapes
: 
?
addAdddnn/logits/BiasAdd:linear/linear_model/linear_model/linear_model/weighted_sum*
T0*'
_output_shapes
:?????????
P
head/predictions/logits/ShapeShapeadd*
_output_shapes
:*
T0
s
1head/predictions/logits/assert_rank_at_least/rankConst*
value	B :*
dtype0*
_output_shapes
: 
c
[head/predictions/logits/assert_rank_at_least/assert_type/statically_determined_correct_typeNoOp
T
Lhead/predictions/logits/assert_rank_at_least/static_checks_determined_all_okNoOp
[
head/predictions/logisticSigmoidadd*
T0*'
_output_shapes
:?????????
_
head/predictions/zeros_like	ZerosLikeadd*
T0*'
_output_shapes
:?????????
q
&head/predictions/two_class_logits/axisConst*
dtype0*
_output_shapes
: *
valueB :
?????????
?
!head/predictions/two_class_logitsConcatV2head/predictions/zeros_likeadd&head/predictions/two_class_logits/axis*
N*'
_output_shapes
:?????????*
T0
~
head/predictions/probabilitiesSoftmax!head/predictions/two_class_logits*
T0*'
_output_shapes
:?????????
o
$head/predictions/class_ids/dimensionConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
head/predictions/class_idsArgMax!head/predictions/two_class_logits$head/predictions/class_ids/dimension*
T0*#
_output_shapes
:?????????
j
head/predictions/ExpandDims/dimConst*
valueB :
?????????*
dtype0*
_output_shapes
: 
?
head/predictions/ExpandDims
ExpandDimshead/predictions/class_idshead/predictions/ExpandDims/dim*
T0	*'
_output_shapes
:?????????
I
head/predictions/ShapeShapeadd*
T0*
_output_shapes
:
n
$head/predictions/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
p
&head/predictions/strided_slice/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
p
&head/predictions/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
head/predictions/strided_sliceStridedSlicehead/predictions/Shape$head/predictions/strided_slice/stack&head/predictions/strided_slice/stack_1&head/predictions/strided_slice/stack_2*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask
^
head/predictions/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
^
head/predictions/range/limitConst*
dtype0*
_output_shapes
: *
value	B :
^
head/predictions/range/deltaConst*
dtype0*
_output_shapes
: *
value	B :
?
head/predictions/rangeRangehead/predictions/range/starthead/predictions/range/limithead/predictions/range/delta*
_output_shapes
:
c
!head/predictions/ExpandDims_1/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
head/predictions/ExpandDims_1
ExpandDimshead/predictions/range!head/predictions/ExpandDims_1/dim*
T0*
_output_shapes

:
c
!head/predictions/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
?
head/predictions/Tile/multiplesPackhead/predictions/strided_slice!head/predictions/Tile/multiples/1*
T0*
N*
_output_shapes
:
?
head/predictions/TileTilehead/predictions/ExpandDims_1head/predictions/Tile/multiples*
T0*'
_output_shapes
:?????????
K
head/predictions/Shape_1Shapeadd*
T0*
_output_shapes
:
p
&head/predictions/strided_slice_1/stackConst*
valueB: *
dtype0*
_output_shapes
:
r
(head/predictions/strided_slice_1/stack_1Const*
valueB:*
dtype0*
_output_shapes
:
r
(head/predictions/strided_slice_1/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
 head/predictions/strided_slice_1StridedSlicehead/predictions/Shape_1&head/predictions/strided_slice_1/stack(head/predictions/strided_slice_1/stack_1(head/predictions/strided_slice_1/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
`
head/predictions/range_1/startConst*
value	B : *
dtype0*
_output_shapes
: 
`
head/predictions/range_1/limitConst*
dtype0*
_output_shapes
: *
value	B :
`
head/predictions/range_1/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
?
head/predictions/range_1Rangehead/predictions/range_1/starthead/predictions/range_1/limithead/predictions/range_1/delta*
_output_shapes
:
d
head/predictions/AsStringAsStringhead/predictions/range_1*
_output_shapes
:*
T0
c
!head/predictions/ExpandDims_2/dimConst*
value	B : *
dtype0*
_output_shapes
: 
?
head/predictions/ExpandDims_2
ExpandDimshead/predictions/AsString!head/predictions/ExpandDims_2/dim*
T0*
_output_shapes

:
e
#head/predictions/Tile_1/multiples/1Const*
dtype0*
_output_shapes
: *
value	B :
?
!head/predictions/Tile_1/multiplesPack head/predictions/strided_slice_1#head/predictions/Tile_1/multiples/1*
T0*
N*
_output_shapes
:
?
head/predictions/Tile_1Tilehead/predictions/ExpandDims_2!head/predictions/Tile_1/multiples*
T0*'
_output_shapes
:?????????
w
head/predictions/str_classesAsStringhead/predictions/ExpandDims*
T0	*'
_output_shapes
:?????????
X

head/ShapeShapehead/predictions/probabilities*
T0*
_output_shapes
:
b
head/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
d
head/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB:
d
head/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
?
head/strided_sliceStridedSlice
head/Shapehead/strided_slice/stackhead/strided_slice/stack_1head/strided_slice/stack_2*
shrink_axis_mask*
T0*
Index0*
_output_shapes
: 
R
head/range/startConst*
value	B : *
dtype0*
_output_shapes
: 
R
head/range/limitConst*
value	B :*
dtype0*
_output_shapes
: 
R
head/range/deltaConst*
value	B :*
dtype0*
_output_shapes
: 
e

head/rangeRangehead/range/starthead/range/limithead/range/delta*
_output_shapes
:
J
head/AsStringAsString
head/range*
_output_shapes
:*
T0
U
head/ExpandDims/dimConst*
value	B : *
dtype0*
_output_shapes
: 
j
head/ExpandDims
ExpandDimshead/AsStringhead/ExpandDims/dim*
T0*
_output_shapes

:
W
head/Tile/multiples/1Const*
value	B :*
dtype0*
_output_shapes
: 
t
head/Tile/multiplesPackhead/strided_slicehead/Tile/multiples/1*
N*
_output_shapes
:*
T0
i
	head/TileTilehead/ExpandDimshead/Tile/multiples*
T0*'
_output_shapes
:?????????

initNoOp
?
init_all_tablesNoOpz^dnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/table_init/LookupTableImportV2t^dnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_table/table_init/LookupTableImportV2?^dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/table_init/LookupTableImportV2p^dnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/hash_table/table_init/LookupTableImportV2?^dnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/table_init/LookupTableImportV2z^dnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/hash_table/table_init/LookupTableImportV2?^linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/hash_table/table_init/LookupTableImportV2s^linear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/table_init/LookupTableImportV2m^linear/linear_model/linear_model/linear_model/gender/gender_lookup/hash_table/table_init/LookupTableImportV2}^linear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/table_init/LookupTableImportV2y^linear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/table_init/LookupTableImportV2s^linear/linear_model/linear_model/linear_model/workclass/workclass_lookup/hash_table/table_init/LookupTableImportV2

init_1NoOp
4

group_depsNoOp^init^init_1^init_all_tables
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
shape: *
dtype0*
_output_shapes
: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
r
save/Read/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0*
dtype0*
_output_shapes
:d
X
save/IdentityIdentitysave/Read/ReadVariableOp*
_output_shapes
:d*
T0
^
save/Identity_1Identitysave/Identity"/device:CPU:0*
T0*
_output_shapes
:d
z
save/Read_1/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0*
dtype0*
_output_shapes

:Bd
`
save/Identity_2Identitysave/Read_1/ReadVariableOp*
T0*
_output_shapes

:Bd
d
save/Identity_3Identitysave/Identity_2"/device:CPU:0*
T0*
_output_shapes

:Bd
t
save/Read_2/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0*
dtype0*
_output_shapes
:F
\
save/Identity_4Identitysave/Read_2/ReadVariableOp*
T0*
_output_shapes
:F
`
save/Identity_5Identitysave/Identity_4"/device:CPU:0*
_output_shapes
:F*
T0
z
save/Read_3/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0*
dtype0*
_output_shapes

:dF
`
save/Identity_6Identitysave/Read_3/ReadVariableOp*
T0*
_output_shapes

:dF
d
save/Identity_7Identitysave/Identity_6"/device:CPU:0*
T0*
_output_shapes

:dF
t
save/Read_4/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/bias/part_0*
dtype0*
_output_shapes
:0
\
save/Identity_8Identitysave/Read_4/ReadVariableOp*
T0*
_output_shapes
:0
`
save/Identity_9Identitysave/Identity_8"/device:CPU:0*
T0*
_output_shapes
:0
z
save/Read_5/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/kernel/part_0*
dtype0*
_output_shapes

:F0
a
save/Identity_10Identitysave/Read_5/ReadVariableOp*
T0*
_output_shapes

:F0
f
save/Identity_11Identitysave/Identity_10"/device:CPU:0*
T0*
_output_shapes

:F0
t
save/Read_6/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/bias/part_0*
dtype0*
_output_shapes
:"
]
save/Identity_12Identitysave/Read_6/ReadVariableOp*
T0*
_output_shapes
:"
b
save/Identity_13Identitysave/Identity_12"/device:CPU:0*
_output_shapes
:"*
T0
z
save/Read_7/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/kernel/part_0*
dtype0*
_output_shapes

:0"
a
save/Identity_14Identitysave/Read_7/ReadVariableOp*
T0*
_output_shapes

:0"
f
save/Identity_15Identitysave/Identity_14"/device:CPU:0*
T0*
_output_shapes

:0"
?
save/Read_8/ReadVariableOpReadVariableOp\dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0*
dtype0*
_output_shapes

:d
a
save/Identity_16Identitysave/Read_8/ReadVariableOp*
T0*
_output_shapes

:d
f
save/Identity_17Identitysave/Identity_16"/device:CPU:0*
T0*
_output_shapes

:d
?
save/Read_9/ReadVariableOpReadVariableOpXdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0*
dtype0*
_output_shapes

:d
a
save/Identity_18Identitysave/Read_9/ReadVariableOp*
_output_shapes

:d*
T0
f
save/Identity_19Identitysave/Identity_18"/device:CPU:0*
T0*
_output_shapes

:d
n
save/Read_10/ReadVariableOpReadVariableOpdnn/logits/bias/part_0*
dtype0*
_output_shapes
:
^
save/Identity_20Identitysave/Read_10/ReadVariableOp*
_output_shapes
:*
T0
b
save/Identity_21Identitysave/Identity_20"/device:CPU:0*
_output_shapes
:*
T0
t
save/Read_11/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0*
dtype0*
_output_shapes

:"
b
save/Identity_22Identitysave/Read_11/ReadVariableOp*
T0*
_output_shapes

:"
f
save/Identity_23Identitysave/Identity_22"/device:CPU:0*
T0*
_output_shapes

:"
?
save/Read_12/ReadVariableOpReadVariableOp1linear/linear_model/age_bucketized/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_24Identitysave/Read_12/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_25Identitysave/Identity_24"/device:CPU:0*
_output_shapes

:*
T0
?
save/Read_13/ReadVariableOpReadVariableOpElinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0*
dtype0* 
_output_shapes
:
??=
d
save/Identity_26Identitysave/Read_13/ReadVariableOp* 
_output_shapes
:
??=*
T0
h
save/Identity_27Identitysave/Identity_26"/device:CPU:0* 
_output_shapes
:
??=*
T0

save/Read_14/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0*
dtype0*
_output_shapes
:
^
save/Identity_28Identitysave/Read_14/ReadVariableOp*
T0*
_output_shapes
:
b
save/Identity_29Identitysave/Identity_28"/device:CPU:0*
T0*
_output_shapes
:
?
save/Read_15/ReadVariableOpReadVariableOp,linear/linear_model/education/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_30Identitysave/Read_15/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_31Identitysave/Identity_30"/device:CPU:0*
_output_shapes

:*
T0
?
save/Read_16/ReadVariableOpReadVariableOp9linear/linear_model/education_X_occupation/weights/part_0*
dtype0*
_output_shapes
:	?N
c
save/Identity_32Identitysave/Read_16/ReadVariableOp*
T0*
_output_shapes
:	?N
g
save/Identity_33Identitysave/Identity_32"/device:CPU:0*
T0*
_output_shapes
:	?N
?
save/Read_17/ReadVariableOpReadVariableOp)linear/linear_model/gender/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_34Identitysave/Read_17/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_35Identitysave/Identity_34"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_18/ReadVariableOpReadVariableOp1linear/linear_model/marital_status/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_36Identitysave/Read_18/ReadVariableOp*
T0*
_output_shapes

:
f
save/Identity_37Identitysave/Identity_36"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_19/ReadVariableOpReadVariableOp1linear/linear_model/native_country/weights/part_0*
dtype0*
_output_shapes

:d
b
save/Identity_38Identitysave/Read_19/ReadVariableOp*
T0*
_output_shapes

:d
f
save/Identity_39Identitysave/Identity_38"/device:CPU:0*
T0*
_output_shapes

:d
?
save/Read_20/ReadVariableOpReadVariableOp>linear/linear_model/native_country_X_occupation/weights/part_0*
dtype0*
_output_shapes
:	?N
c
save/Identity_40Identitysave/Read_20/ReadVariableOp*
T0*
_output_shapes
:	?N
g
save/Identity_41Identitysave/Identity_40"/device:CPU:0*
T0*
_output_shapes
:	?N
?
save/Read_21/ReadVariableOpReadVariableOp-linear/linear_model/occupation/weights/part_0*
dtype0*
_output_shapes

:d
b
save/Identity_42Identitysave/Read_21/ReadVariableOp*
_output_shapes

:d*
T0
f
save/Identity_43Identitysave/Identity_42"/device:CPU:0*
T0*
_output_shapes

:d
?
save/Read_22/ReadVariableOpReadVariableOp/linear/linear_model/relationship/weights/part_0*
dtype0*
_output_shapes

:
b
save/Identity_44Identitysave/Read_22/ReadVariableOp*
_output_shapes

:*
T0
f
save/Identity_45Identitysave/Identity_44"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_23/ReadVariableOpReadVariableOp,linear/linear_model/workclass/weights/part_0*
dtype0*
_output_shapes

:	
b
save/Identity_46Identitysave/Read_23/ReadVariableOp*
T0*
_output_shapes

:	
f
save/Identity_47Identitysave/Identity_46"/device:CPU:0*
T0*
_output_shapes

:	
?
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_ed19f30bf3854066b8f7c709e2359c13/part*
dtype0*
_output_shapes
: 
d
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: 
Q
save/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
k
save/ShardedFilename/shardConst"/device:CPU:0*
value	B : *
dtype0*
_output_shapes
: 
?
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
{
save/SaveV2/tensor_namesConst"/device:CPU:0* 
valueBBglobal_step*
dtype0*
_output_shapes
:
t
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
?
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesglobal_step"/device:CPU:0*
dtypes
2	
?
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
m
save/ShardedFilename_1/shardConst"/device:CPU:0*
value	B :*
dtype0*
_output_shapes
: 
?
save/ShardedFilename_1ShardedFilenamesave/StringJoinsave/ShardedFilename_1/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
?
save/Read_24/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:d
m
save/Identity_48Identitysave/Read_24/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:d
b
save/Identity_49Identitysave/Identity_48"/device:CPU:0*
T0*
_output_shapes
:d
?
save/Read_25/ReadVariableOpReadVariableOpdnn/hiddenlayer_0/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:Bd
q
save/Identity_50Identitysave/Read_25/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:Bd
f
save/Identity_51Identitysave/Identity_50"/device:CPU:0*
T0*
_output_shapes

:Bd
?
save/Read_26/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:F
m
save/Identity_52Identitysave/Read_26/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:F
b
save/Identity_53Identitysave/Identity_52"/device:CPU:0*
T0*
_output_shapes
:F
?
save/Read_27/ReadVariableOpReadVariableOpdnn/hiddenlayer_1/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:dF
q
save/Identity_54Identitysave/Read_27/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:dF
f
save/Identity_55Identitysave/Identity_54"/device:CPU:0*
T0*
_output_shapes

:dF
?
save/Read_28/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:0
m
save/Identity_56Identitysave/Read_28/ReadVariableOp"/device:CPU:0*
_output_shapes
:0*
T0
b
save/Identity_57Identitysave/Identity_56"/device:CPU:0*
T0*
_output_shapes
:0
?
save/Read_29/ReadVariableOpReadVariableOpdnn/hiddenlayer_2/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:F0
q
save/Identity_58Identitysave/Read_29/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:F0
f
save/Identity_59Identitysave/Identity_58"/device:CPU:0*
_output_shapes

:F0*
T0
?
save/Read_30/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:"
m
save/Identity_60Identitysave/Read_30/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:"
b
save/Identity_61Identitysave/Identity_60"/device:CPU:0*
T0*
_output_shapes
:"
?
save/Read_31/ReadVariableOpReadVariableOpdnn/hiddenlayer_3/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:0"
q
save/Identity_62Identitysave/Read_31/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:0"
f
save/Identity_63Identitysave/Identity_62"/device:CPU:0*
T0*
_output_shapes

:0"
?
save/Read_32/ReadVariableOpReadVariableOp\dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:d
q
save/Identity_64Identitysave/Read_32/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:d
f
save/Identity_65Identitysave/Identity_64"/device:CPU:0*
T0*
_output_shapes

:d
?
save/Read_33/ReadVariableOpReadVariableOpXdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:d
q
save/Identity_66Identitysave/Read_33/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:d
f
save/Identity_67Identitysave/Identity_66"/device:CPU:0*
T0*
_output_shapes

:d
}
save/Read_34/ReadVariableOpReadVariableOpdnn/logits/bias/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_68Identitysave/Read_34/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_69Identitysave/Identity_68"/device:CPU:0*
_output_shapes
:*
T0
?
save/Read_35/ReadVariableOpReadVariableOpdnn/logits/kernel/part_0"/device:CPU:0*
dtype0*
_output_shapes

:"
q
save/Identity_70Identitysave/Read_35/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:"
f
save/Identity_71Identitysave/Identity_70"/device:CPU:0*
T0*
_output_shapes

:"
?
save/Read_36/ReadVariableOpReadVariableOp1linear/linear_model/age_bucketized/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_72Identitysave/Read_36/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_73Identitysave/Identity_72"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_37/ReadVariableOpReadVariableOpElinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0"/device:CPU:0*
dtype0* 
_output_shapes
:
??=
s
save/Identity_74Identitysave/Read_37/ReadVariableOp"/device:CPU:0*
T0* 
_output_shapes
:
??=
h
save/Identity_75Identitysave/Identity_74"/device:CPU:0*
T0* 
_output_shapes
:
??=
?
save/Read_38/ReadVariableOpReadVariableOp'linear/linear_model/bias_weights/part_0"/device:CPU:0*
dtype0*
_output_shapes
:
m
save/Identity_76Identitysave/Read_38/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:
b
save/Identity_77Identitysave/Identity_76"/device:CPU:0*
T0*
_output_shapes
:
?
save/Read_39/ReadVariableOpReadVariableOp,linear/linear_model/education/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_78Identitysave/Read_39/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_79Identitysave/Identity_78"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_40/ReadVariableOpReadVariableOp9linear/linear_model/education_X_occupation/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes
:	?N
r
save/Identity_80Identitysave/Read_40/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:	?N
g
save/Identity_81Identitysave/Identity_80"/device:CPU:0*
_output_shapes
:	?N*
T0
?
save/Read_41/ReadVariableOpReadVariableOp)linear/linear_model/gender/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_82Identitysave/Read_41/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_83Identitysave/Identity_82"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_42/ReadVariableOpReadVariableOp1linear/linear_model/marital_status/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_84Identitysave/Read_42/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:
f
save/Identity_85Identitysave/Identity_84"/device:CPU:0*
T0*
_output_shapes

:
?
save/Read_43/ReadVariableOpReadVariableOp1linear/linear_model/native_country/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:d
q
save/Identity_86Identitysave/Read_43/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:d
f
save/Identity_87Identitysave/Identity_86"/device:CPU:0*
T0*
_output_shapes

:d
?
save/Read_44/ReadVariableOpReadVariableOp>linear/linear_model/native_country_X_occupation/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes
:	?N
r
save/Identity_88Identitysave/Read_44/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes
:	?N
g
save/Identity_89Identitysave/Identity_88"/device:CPU:0*
T0*
_output_shapes
:	?N
?
save/Read_45/ReadVariableOpReadVariableOp-linear/linear_model/occupation/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:d
q
save/Identity_90Identitysave/Read_45/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:d
f
save/Identity_91Identitysave/Identity_90"/device:CPU:0*
T0*
_output_shapes

:d
?
save/Read_46/ReadVariableOpReadVariableOp/linear/linear_model/relationship/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:
q
save/Identity_92Identitysave/Read_46/ReadVariableOp"/device:CPU:0*
_output_shapes

:*
T0
f
save/Identity_93Identitysave/Identity_92"/device:CPU:0*
_output_shapes

:*
T0
?
save/Read_47/ReadVariableOpReadVariableOp,linear/linear_model/workclass/weights/part_0"/device:CPU:0*
dtype0*
_output_shapes

:	
q
save/Identity_94Identitysave/Read_47/ReadVariableOp"/device:CPU:0*
T0*
_output_shapes

:	
f
save/Identity_95Identitysave/Identity_94"/device:CPU:0*
T0*
_output_shapes

:	
?
save/SaveV2_1/tensor_namesConst"/device:CPU:0*?
value?B?Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/hiddenlayer_3/biasBdnn/hiddenlayer_3/kernelBUdnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weightsBQdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weightsBdnn/logits/biasBdnn/logits/kernelB*linear/linear_model/age_bucketized/weightsB>linear/linear_model/age_bucketized_X_occupation_X_race/weightsB linear/linear_model/bias_weightsB%linear/linear_model/education/weightsB2linear/linear_model/education_X_occupation/weightsB"linear/linear_model/gender/weightsB*linear/linear_model/marital_status/weightsB*linear/linear_model/native_country/weightsB7linear/linear_model/native_country_X_occupation/weightsB&linear/linear_model/occupation/weightsB(linear/linear_model/relationship/weightsB%linear/linear_model/workclass/weights*
dtype0*
_output_shapes
:
?
save/SaveV2_1/shape_and_slicesConst"/device:CPU:0*?
value?B?B	100 0,100B66 100 0,66:0,100B70 0,70B100 70 0,100:0,70B48 0,48B70 48 0,70:0,48B34 0,34B48 34 0,48:0,34B100 8 0,100:0,8B100 8 0,100:0,8B1 0,1B34 1 0,34:0,1B11 1 0,11:0,1B1000000 1 0,1000000:0,1B1 0,1B16 1 0,16:0,1B10000 1 0,10000:0,1B2 1 0,2:0,1B7 1 0,7:0,1B100 1 0,100:0,1B10000 1 0,10000:0,1B100 1 0,100:0,1B6 1 0,6:0,1B9 1 0,9:0,1*
dtype0*
_output_shapes
:
?
save/SaveV2_1SaveV2save/ShardedFilename_1save/SaveV2_1/tensor_namessave/SaveV2_1/shape_and_slicessave/Identity_49save/Identity_51save/Identity_53save/Identity_55save/Identity_57save/Identity_59save/Identity_61save/Identity_63save/Identity_65save/Identity_67save/Identity_69save/Identity_71save/Identity_73save/Identity_75save/Identity_77save/Identity_79save/Identity_81save/Identity_83save/Identity_85save/Identity_87save/Identity_89save/Identity_91save/Identity_93save/Identity_95"/device:CPU:0*&
dtypes
2
?
save/control_dependency_1Identitysave/ShardedFilename_1^save/SaveV2_1"/device:CPU:0*
T0*)
_class
loc:@save/ShardedFilename_1*
_output_shapes
: 
?
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilenamesave/ShardedFilename_1^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
N*
_output_shapes
:
u
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0
?
save/Identity_96Identity
save/Const^save/MergeV2Checkpoints^save/control_dependency^save/control_dependency_1"/device:CPU:0*
T0*
_output_shapes
: 
~
save/RestoreV2/tensor_namesConst"/device:CPU:0*
dtype0*
_output_shapes
:* 
valueBBglobal_step
w
save/RestoreV2/shape_and_slicesConst"/device:CPU:0*
valueB
B *
dtype0*
_output_shapes
:
?
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices"/device:CPU:0*
_output_shapes
:*
dtypes
2	
s
save/AssignAssignglobal_stepsave/RestoreV2*
_output_shapes
: *
T0	*
_class
loc:@global_step
(
save/restore_shardNoOp^save/Assign
?
save/RestoreV2_1/tensor_namesConst"/device:CPU:0*?
value?B?Bdnn/hiddenlayer_0/biasBdnn/hiddenlayer_0/kernelBdnn/hiddenlayer_1/biasBdnn/hiddenlayer_1/kernelBdnn/hiddenlayer_2/biasBdnn/hiddenlayer_2/kernelBdnn/hiddenlayer_3/biasBdnn/hiddenlayer_3/kernelBUdnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weightsBQdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weightsBdnn/logits/biasBdnn/logits/kernelB*linear/linear_model/age_bucketized/weightsB>linear/linear_model/age_bucketized_X_occupation_X_race/weightsB linear/linear_model/bias_weightsB%linear/linear_model/education/weightsB2linear/linear_model/education_X_occupation/weightsB"linear/linear_model/gender/weightsB*linear/linear_model/marital_status/weightsB*linear/linear_model/native_country/weightsB7linear/linear_model/native_country_X_occupation/weightsB&linear/linear_model/occupation/weightsB(linear/linear_model/relationship/weightsB%linear/linear_model/workclass/weights*
dtype0*
_output_shapes
:
?
!save/RestoreV2_1/shape_and_slicesConst"/device:CPU:0*?
value?B?B	100 0,100B66 100 0,66:0,100B70 0,70B100 70 0,100:0,70B48 0,48B70 48 0,70:0,48B34 0,34B48 34 0,48:0,34B100 8 0,100:0,8B100 8 0,100:0,8B1 0,1B34 1 0,34:0,1B11 1 0,11:0,1B1000000 1 0,1000000:0,1B1 0,1B16 1 0,16:0,1B10000 1 0,10000:0,1B2 1 0,2:0,1B7 1 0,7:0,1B100 1 0,100:0,1B10000 1 0,10000:0,1B100 1 0,100:0,1B6 1 0,6:0,1B9 1 0,9:0,1*
dtype0*
_output_shapes
:
?
save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices"/device:CPU:0*?
_output_shapes?
?:d:Bd:F:dF:0:F0:":0":d:d::"::
??=:::	?N:::d:	?N:d::	*&
dtypes
2
b
save/Identity_97Identitysave/RestoreV2_1"/device:CPU:0*
T0*
_output_shapes
:d
v
save/AssignVariableOpAssignVariableOpdnn/hiddenlayer_0/bias/part_0save/Identity_97"/device:CPU:0*
dtype0
h
save/Identity_98Identitysave/RestoreV2_1:1"/device:CPU:0*
T0*
_output_shapes

:Bd
z
save/AssignVariableOp_1AssignVariableOpdnn/hiddenlayer_0/kernel/part_0save/Identity_98"/device:CPU:0*
dtype0
d
save/Identity_99Identitysave/RestoreV2_1:2"/device:CPU:0*
T0*
_output_shapes
:F
x
save/AssignVariableOp_2AssignVariableOpdnn/hiddenlayer_1/bias/part_0save/Identity_99"/device:CPU:0*
dtype0
i
save/Identity_100Identitysave/RestoreV2_1:3"/device:CPU:0*
T0*
_output_shapes

:dF
{
save/AssignVariableOp_3AssignVariableOpdnn/hiddenlayer_1/kernel/part_0save/Identity_100"/device:CPU:0*
dtype0
e
save/Identity_101Identitysave/RestoreV2_1:4"/device:CPU:0*
T0*
_output_shapes
:0
y
save/AssignVariableOp_4AssignVariableOpdnn/hiddenlayer_2/bias/part_0save/Identity_101"/device:CPU:0*
dtype0
i
save/Identity_102Identitysave/RestoreV2_1:5"/device:CPU:0*
T0*
_output_shapes

:F0
{
save/AssignVariableOp_5AssignVariableOpdnn/hiddenlayer_2/kernel/part_0save/Identity_102"/device:CPU:0*
dtype0
e
save/Identity_103Identitysave/RestoreV2_1:6"/device:CPU:0*
T0*
_output_shapes
:"
y
save/AssignVariableOp_6AssignVariableOpdnn/hiddenlayer_3/bias/part_0save/Identity_103"/device:CPU:0*
dtype0
i
save/Identity_104Identitysave/RestoreV2_1:7"/device:CPU:0*
T0*
_output_shapes

:0"
{
save/AssignVariableOp_7AssignVariableOpdnn/hiddenlayer_3/kernel/part_0save/Identity_104"/device:CPU:0*
dtype0
i
save/Identity_105Identitysave/RestoreV2_1:8"/device:CPU:0*
T0*
_output_shapes

:d
?
save/AssignVariableOp_8AssignVariableOp\dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0save/Identity_105"/device:CPU:0*
dtype0
i
save/Identity_106Identitysave/RestoreV2_1:9"/device:CPU:0*
T0*
_output_shapes

:d
?
save/AssignVariableOp_9AssignVariableOpXdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0save/Identity_106"/device:CPU:0*
dtype0
f
save/Identity_107Identitysave/RestoreV2_1:10"/device:CPU:0*
_output_shapes
:*
T0
s
save/AssignVariableOp_10AssignVariableOpdnn/logits/bias/part_0save/Identity_107"/device:CPU:0*
dtype0
j
save/Identity_108Identitysave/RestoreV2_1:11"/device:CPU:0*
T0*
_output_shapes

:"
u
save/AssignVariableOp_11AssignVariableOpdnn/logits/kernel/part_0save/Identity_108"/device:CPU:0*
dtype0
j
save/Identity_109Identitysave/RestoreV2_1:12"/device:CPU:0*
T0*
_output_shapes

:
?
save/AssignVariableOp_12AssignVariableOp1linear/linear_model/age_bucketized/weights/part_0save/Identity_109"/device:CPU:0*
dtype0
l
save/Identity_110Identitysave/RestoreV2_1:13"/device:CPU:0* 
_output_shapes
:
??=*
T0
?
save/AssignVariableOp_13AssignVariableOpElinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0save/Identity_110"/device:CPU:0*
dtype0
f
save/Identity_111Identitysave/RestoreV2_1:14"/device:CPU:0*
T0*
_output_shapes
:
?
save/AssignVariableOp_14AssignVariableOp'linear/linear_model/bias_weights/part_0save/Identity_111"/device:CPU:0*
dtype0
j
save/Identity_112Identitysave/RestoreV2_1:15"/device:CPU:0*
T0*
_output_shapes

:
?
save/AssignVariableOp_15AssignVariableOp,linear/linear_model/education/weights/part_0save/Identity_112"/device:CPU:0*
dtype0
k
save/Identity_113Identitysave/RestoreV2_1:16"/device:CPU:0*
T0*
_output_shapes
:	?N
?
save/AssignVariableOp_16AssignVariableOp9linear/linear_model/education_X_occupation/weights/part_0save/Identity_113"/device:CPU:0*
dtype0
j
save/Identity_114Identitysave/RestoreV2_1:17"/device:CPU:0*
_output_shapes

:*
T0
?
save/AssignVariableOp_17AssignVariableOp)linear/linear_model/gender/weights/part_0save/Identity_114"/device:CPU:0*
dtype0
j
save/Identity_115Identitysave/RestoreV2_1:18"/device:CPU:0*
T0*
_output_shapes

:
?
save/AssignVariableOp_18AssignVariableOp1linear/linear_model/marital_status/weights/part_0save/Identity_115"/device:CPU:0*
dtype0
j
save/Identity_116Identitysave/RestoreV2_1:19"/device:CPU:0*
T0*
_output_shapes

:d
?
save/AssignVariableOp_19AssignVariableOp1linear/linear_model/native_country/weights/part_0save/Identity_116"/device:CPU:0*
dtype0
k
save/Identity_117Identitysave/RestoreV2_1:20"/device:CPU:0*
T0*
_output_shapes
:	?N
?
save/AssignVariableOp_20AssignVariableOp>linear/linear_model/native_country_X_occupation/weights/part_0save/Identity_117"/device:CPU:0*
dtype0
j
save/Identity_118Identitysave/RestoreV2_1:21"/device:CPU:0*
T0*
_output_shapes

:d
?
save/AssignVariableOp_21AssignVariableOp-linear/linear_model/occupation/weights/part_0save/Identity_118"/device:CPU:0*
dtype0
j
save/Identity_119Identitysave/RestoreV2_1:22"/device:CPU:0*
T0*
_output_shapes

:
?
save/AssignVariableOp_22AssignVariableOp/linear/linear_model/relationship/weights/part_0save/Identity_119"/device:CPU:0*
dtype0
j
save/Identity_120Identitysave/RestoreV2_1:23"/device:CPU:0*
T0*
_output_shapes

:	
?
save/AssignVariableOp_23AssignVariableOp,linear/linear_model/workclass/weights/part_0save/Identity_120"/device:CPU:0*
dtype0
?
save/restore_shard_1NoOp^save/AssignVariableOp^save/AssignVariableOp_1^save/AssignVariableOp_10^save/AssignVariableOp_11^save/AssignVariableOp_12^save/AssignVariableOp_13^save/AssignVariableOp_14^save/AssignVariableOp_15^save/AssignVariableOp_16^save/AssignVariableOp_17^save/AssignVariableOp_18^save/AssignVariableOp_19^save/AssignVariableOp_2^save/AssignVariableOp_20^save/AssignVariableOp_21^save/AssignVariableOp_22^save/AssignVariableOp_23^save/AssignVariableOp_3^save/AssignVariableOp_4^save/AssignVariableOp_5^save/AssignVariableOp_6^save/AssignVariableOp_7^save/AssignVariableOp_8^save/AssignVariableOp_9"/device:CPU:0
2
save/restore_all/NoOpNoOp^save/restore_shard
E
save/restore_all/NoOp_1NoOp^save/restore_shard_1"/device:CPU:0
J
save/restore_allNoOp^save/restore_all/NoOp^save/restore_all/NoOp_1"&?
save/Const:0save/Identity_96:0save/restore_all (5 @F8"??
cond_context????
?
 dnn/zero_fraction/cond/cond_text dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_t:0 *?
dnn/hiddenlayer_0/Relu:0
dnn/zero_fraction/cond/Cast:0
+dnn/zero_fraction/cond/count_nonzero/Cast:0
,dnn/zero_fraction/cond/count_nonzero/Const:0
6dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
/dnn/zero_fraction/cond/count_nonzero/NotEqual:0
4dnn/zero_fraction/cond/count_nonzero/nonzero_count:0
,dnn/zero_fraction/cond/count_nonzero/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_t:0R
dnn/hiddenlayer_0/Relu:06dnn/zero_fraction/cond/count_nonzero/NotEqual/Switch:1D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0
?
"dnn/zero_fraction/cond/cond_text_1 dnn/zero_fraction/cond/pred_id:0!dnn/zero_fraction/cond/switch_f:0*?
dnn/hiddenlayer_0/Relu:0
-dnn/zero_fraction/cond/count_nonzero_1/Cast:0
.dnn/zero_fraction/cond/count_nonzero_1/Const:0
8dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
1dnn/zero_fraction/cond/count_nonzero_1/NotEqual:0
6dnn/zero_fraction/cond/count_nonzero_1/nonzero_count:0
.dnn/zero_fraction/cond/count_nonzero_1/zeros:0
 dnn/zero_fraction/cond/pred_id:0
!dnn/zero_fraction/cond/switch_f:0D
 dnn/zero_fraction/cond/pred_id:0 dnn/zero_fraction/cond/pred_id:0T
dnn/hiddenlayer_0/Relu:08dnn/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
?
"dnn/zero_fraction_1/cond/cond_text"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_t:0 *?
dnn/hiddenlayer_1/Relu:0
dnn/zero_fraction_1/cond/Cast:0
-dnn/zero_fraction_1/cond/count_nonzero/Cast:0
.dnn/zero_fraction_1/cond/count_nonzero/Const:0
8dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_1/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_1/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_1/cond/count_nonzero/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_t:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0T
dnn/hiddenlayer_1/Relu:08dnn/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
?
$dnn/zero_fraction_1/cond/cond_text_1"dnn/zero_fraction_1/cond/pred_id:0#dnn/zero_fraction_1/cond/switch_f:0*?
dnn/hiddenlayer_1/Relu:0
/dnn/zero_fraction_1/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_1/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_1/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_1/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_1/cond/pred_id:0
#dnn/zero_fraction_1/cond/switch_f:0V
dnn/hiddenlayer_1/Relu:0:dnn/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0H
"dnn/zero_fraction_1/cond/pred_id:0"dnn/zero_fraction_1/cond/pred_id:0
?
"dnn/zero_fraction_2/cond/cond_text"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_t:0 *?
dnn/hiddenlayer_2/Relu:0
dnn/zero_fraction_2/cond/Cast:0
-dnn/zero_fraction_2/cond/count_nonzero/Cast:0
.dnn/zero_fraction_2/cond/count_nonzero/Const:0
8dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_2/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_2/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_2/cond/count_nonzero/zeros:0
"dnn/zero_fraction_2/cond/pred_id:0
#dnn/zero_fraction_2/cond/switch_t:0H
"dnn/zero_fraction_2/cond/pred_id:0"dnn/zero_fraction_2/cond/pred_id:0T
dnn/hiddenlayer_2/Relu:08dnn/zero_fraction_2/cond/count_nonzero/NotEqual/Switch:1
?
$dnn/zero_fraction_2/cond/cond_text_1"dnn/zero_fraction_2/cond/pred_id:0#dnn/zero_fraction_2/cond/switch_f:0*?
dnn/hiddenlayer_2/Relu:0
/dnn/zero_fraction_2/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_2/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_2/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_2/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_2/cond/pred_id:0
#dnn/zero_fraction_2/cond/switch_f:0H
"dnn/zero_fraction_2/cond/pred_id:0"dnn/zero_fraction_2/cond/pred_id:0V
dnn/hiddenlayer_2/Relu:0:dnn/zero_fraction_2/cond/count_nonzero_1/NotEqual/Switch:0
?
"dnn/zero_fraction_3/cond/cond_text"dnn/zero_fraction_3/cond/pred_id:0#dnn/zero_fraction_3/cond/switch_t:0 *?
dnn/hiddenlayer_3/Relu:0
dnn/zero_fraction_3/cond/Cast:0
-dnn/zero_fraction_3/cond/count_nonzero/Cast:0
.dnn/zero_fraction_3/cond/count_nonzero/Const:0
8dnn/zero_fraction_3/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_3/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_3/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_3/cond/count_nonzero/zeros:0
"dnn/zero_fraction_3/cond/pred_id:0
#dnn/zero_fraction_3/cond/switch_t:0H
"dnn/zero_fraction_3/cond/pred_id:0"dnn/zero_fraction_3/cond/pred_id:0T
dnn/hiddenlayer_3/Relu:08dnn/zero_fraction_3/cond/count_nonzero/NotEqual/Switch:1
?
$dnn/zero_fraction_3/cond/cond_text_1"dnn/zero_fraction_3/cond/pred_id:0#dnn/zero_fraction_3/cond/switch_f:0*?
dnn/hiddenlayer_3/Relu:0
/dnn/zero_fraction_3/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_3/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_3/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_3/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_3/cond/pred_id:0
#dnn/zero_fraction_3/cond/switch_f:0H
"dnn/zero_fraction_3/cond/pred_id:0"dnn/zero_fraction_3/cond/pred_id:0V
dnn/hiddenlayer_3/Relu:0:dnn/zero_fraction_3/cond/count_nonzero_1/NotEqual/Switch:0
?
"dnn/zero_fraction_4/cond/cond_text"dnn/zero_fraction_4/cond/pred_id:0#dnn/zero_fraction_4/cond/switch_t:0 *?
dnn/logits/BiasAdd:0
dnn/zero_fraction_4/cond/Cast:0
-dnn/zero_fraction_4/cond/count_nonzero/Cast:0
.dnn/zero_fraction_4/cond/count_nonzero/Const:0
8dnn/zero_fraction_4/cond/count_nonzero/NotEqual/Switch:1
1dnn/zero_fraction_4/cond/count_nonzero/NotEqual:0
6dnn/zero_fraction_4/cond/count_nonzero/nonzero_count:0
.dnn/zero_fraction_4/cond/count_nonzero/zeros:0
"dnn/zero_fraction_4/cond/pred_id:0
#dnn/zero_fraction_4/cond/switch_t:0H
"dnn/zero_fraction_4/cond/pred_id:0"dnn/zero_fraction_4/cond/pred_id:0P
dnn/logits/BiasAdd:08dnn/zero_fraction_4/cond/count_nonzero/NotEqual/Switch:1
?
$dnn/zero_fraction_4/cond/cond_text_1"dnn/zero_fraction_4/cond/pred_id:0#dnn/zero_fraction_4/cond/switch_f:0*?
dnn/logits/BiasAdd:0
/dnn/zero_fraction_4/cond/count_nonzero_1/Cast:0
0dnn/zero_fraction_4/cond/count_nonzero_1/Const:0
:dnn/zero_fraction_4/cond/count_nonzero_1/NotEqual/Switch:0
3dnn/zero_fraction_4/cond/count_nonzero_1/NotEqual:0
8dnn/zero_fraction_4/cond/count_nonzero_1/nonzero_count:0
0dnn/zero_fraction_4/cond/count_nonzero_1/zeros:0
"dnn/zero_fraction_4/cond/pred_id:0
#dnn/zero_fraction_4/cond/switch_f:0H
"dnn/zero_fraction_4/cond/pred_id:0"dnn/zero_fraction_4/cond/pred_id:0R
dnn/logits/BiasAdd:0:dnn/zero_fraction_4/cond/count_nonzero_1/NotEqual/Switch:0
?
4linear/zero_fraction/total_zero/zero_count/cond_text4linear/zero_fraction/total_zero/zero_count/pred_id:05linear/zero_fraction/total_zero/zero_count/switch_t:0 *?
2linear/zero_fraction/total_zero/zero_count/Const:0
4linear/zero_fraction/total_zero/zero_count/pred_id:0
5linear/zero_fraction/total_zero/zero_count/switch_t:0l
4linear/zero_fraction/total_zero/zero_count/pred_id:04linear/zero_fraction/total_zero/zero_count/pred_id:0
?.
6linear/zero_fraction/total_zero/zero_count/cond_text_14linear/zero_fraction/total_zero/zero_count/pred_id:05linear/zero_fraction/total_zero/zero_count/switch_f:0*?
3linear/linear_model/age_bucketized/weights/part_0:0
&linear/zero_fraction/total_size/Size:0
;linear/zero_fraction/total_zero/zero_count/ToFloat/Switch:0
4linear/zero_fraction/total_zero/zero_count/ToFloat:0
0linear/zero_fraction/total_zero/zero_count/mul:0
4linear/zero_fraction/total_zero/zero_count/pred_id:0
5linear/zero_fraction/total_zero/zero_count/switch_f:0
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual/y:0
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/LessEqual:0
Plinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/Switch:0
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0
?linear/zero_fraction/total_zero/zero_count/zero_fraction/Size:0
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Cast:0
Elinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Merge:0
Elinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Merge:1
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch:0
Flinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Switch:1
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Cast:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Const:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Vlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual:0
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_count:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zeros:0
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Cast:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Const:0
_linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Xlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zeros:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t:0
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast:0
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/Cast_1:0
Qlinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/sub:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/counts_to_fraction/truediv:0
Clinear/zero_fraction/total_zero/zero_count/zero_fraction/fraction:0e
&linear/zero_fraction/total_size/Size:0;linear/zero_fraction/total_zero/zero_count/ToFloat/Switch:0?
3linear/linear_model/age_bucketized/weights/part_0:0Plinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp/Switch:0l
4linear/zero_fraction/total_zero/zero_count/pred_id:04linear/zero_fraction/total_zero/zero_count/pred_id:02?

?

Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/cond_textGlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t:0 *?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0
Dlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/Cast:0
Rlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Cast:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/Const:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Vlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual:0
[linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/nonzero_count:0
Slinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/zeros:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_t:0?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:02?

?

Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/cond_text_1Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f:0*?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0
Tlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Cast:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/Const:0
_linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Xlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual:0
]linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/zeros:0
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
Hlinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/switch_f:0?
Ilinear/zero_fraction/total_zero/zero_count/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0Glinear/zero_fraction/total_zero/zero_count/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_1/cond_text6linear/zero_fraction/total_zero/zero_count_1/pred_id:07linear/zero_fraction/total_zero/zero_count_1/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_1/Const:0
6linear/zero_fraction/total_zero/zero_count_1/pred_id:0
7linear/zero_fraction/total_zero/zero_count_1/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_1/pred_id:06linear/zero_fraction/total_zero/zero_count_1/pred_id:0
?0
8linear/zero_fraction/total_zero/zero_count_1/cond_text_16linear/zero_fraction/total_zero/zero_count_1/pred_id:07linear/zero_fraction/total_zero/zero_count_1/switch_f:0*?
Glinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0:0
(linear/zero_fraction/total_size/Size_1:0
=linear/zero_fraction/total_zero/zero_count_1/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_1/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_1/mul:0
6linear/zero_fraction/total_zero/zero_count_1/pred_id:0
7linear/zero_fraction/total_zero/zero_count_1/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_1/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_1/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_1/pred_id:06linear/zero_fraction/total_zero/zero_count_1/pred_id:0?
Glinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp/Switch:0i
(linear/zero_fraction/total_size/Size_1:0=linear/zero_fraction/total_zero/zero_count_1/ToFloat/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_1/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_1/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_2/cond_text6linear/zero_fraction/total_zero/zero_count_2/pred_id:07linear/zero_fraction/total_zero/zero_count_2/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_2/Const:0
6linear/zero_fraction/total_zero/zero_count_2/pred_id:0
7linear/zero_fraction/total_zero/zero_count_2/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_2/pred_id:06linear/zero_fraction/total_zero/zero_count_2/pred_id:0
?0
8linear/zero_fraction/total_zero/zero_count_2/cond_text_16linear/zero_fraction/total_zero/zero_count_2/pred_id:07linear/zero_fraction/total_zero/zero_count_2/switch_f:0*?
.linear/linear_model/education/weights/part_0:0
(linear/zero_fraction/total_size/Size_2:0
=linear/zero_fraction/total_zero/zero_count_2/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_2/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_2/mul:0
6linear/zero_fraction/total_zero/zero_count_2/pred_id:0
7linear/zero_fraction/total_zero/zero_count_2/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_2/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_2/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_2/pred_id:06linear/zero_fraction/total_zero/zero_count_2/pred_id:0?
.linear/linear_model/education/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp/Switch:0i
(linear/zero_fraction/total_size/Size_2:0=linear/zero_fraction/total_zero/zero_count_2/ToFloat/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_2/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_2/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_3/cond_text6linear/zero_fraction/total_zero/zero_count_3/pred_id:07linear/zero_fraction/total_zero/zero_count_3/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_3/Const:0
6linear/zero_fraction/total_zero/zero_count_3/pred_id:0
7linear/zero_fraction/total_zero/zero_count_3/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_3/pred_id:06linear/zero_fraction/total_zero/zero_count_3/pred_id:0
?0
8linear/zero_fraction/total_zero/zero_count_3/cond_text_16linear/zero_fraction/total_zero/zero_count_3/pred_id:07linear/zero_fraction/total_zero/zero_count_3/switch_f:0*?
;linear/linear_model/education_X_occupation/weights/part_0:0
(linear/zero_fraction/total_size/Size_3:0
=linear/zero_fraction/total_zero/zero_count_3/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_3/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_3/mul:0
6linear/zero_fraction/total_zero/zero_count_3/pred_id:0
7linear/zero_fraction/total_zero/zero_count_3/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_3/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_3/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_3/pred_id:06linear/zero_fraction/total_zero/zero_count_3/pred_id:0?
;linear/linear_model/education_X_occupation/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp/Switch:0i
(linear/zero_fraction/total_size/Size_3:0=linear/zero_fraction/total_zero/zero_count_3/ToFloat/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_3/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_3/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_4/cond_text6linear/zero_fraction/total_zero/zero_count_4/pred_id:07linear/zero_fraction/total_zero/zero_count_4/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_4/Const:0
6linear/zero_fraction/total_zero/zero_count_4/pred_id:0
7linear/zero_fraction/total_zero/zero_count_4/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_4/pred_id:06linear/zero_fraction/total_zero/zero_count_4/pred_id:0
?/
8linear/zero_fraction/total_zero/zero_count_4/cond_text_16linear/zero_fraction/total_zero/zero_count_4/pred_id:07linear/zero_fraction/total_zero/zero_count_4/switch_f:0*?
+linear/linear_model/gender/weights/part_0:0
(linear/zero_fraction/total_size/Size_4:0
=linear/zero_fraction/total_zero/zero_count_4/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_4/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_4/mul:0
6linear/zero_fraction/total_zero/zero_count_4/pred_id:0
7linear/zero_fraction/total_zero/zero_count_4/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_4/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_4/pred_id:06linear/zero_fraction/total_zero/zero_count_4/pred_id:0?
+linear/linear_model/gender/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp/Switch:0i
(linear/zero_fraction/total_size/Size_4:0=linear/zero_fraction/total_zero/zero_count_4/ToFloat/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_4/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_4/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_5/cond_text6linear/zero_fraction/total_zero/zero_count_5/pred_id:07linear/zero_fraction/total_zero/zero_count_5/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_5/Const:0
6linear/zero_fraction/total_zero/zero_count_5/pred_id:0
7linear/zero_fraction/total_zero/zero_count_5/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_5/pred_id:06linear/zero_fraction/total_zero/zero_count_5/pred_id:0
?0
8linear/zero_fraction/total_zero/zero_count_5/cond_text_16linear/zero_fraction/total_zero/zero_count_5/pred_id:07linear/zero_fraction/total_zero/zero_count_5/switch_f:0*?
3linear/linear_model/marital_status/weights/part_0:0
(linear/zero_fraction/total_size/Size_5:0
=linear/zero_fraction/total_zero/zero_count_5/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_5/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_5/mul:0
6linear/zero_fraction/total_zero/zero_count_5/pred_id:0
7linear/zero_fraction/total_zero/zero_count_5/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_5/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_5/zero_fraction/fraction:0?
3linear/linear_model/marital_status/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp/Switch:0i
(linear/zero_fraction/total_size/Size_5:0=linear/zero_fraction/total_zero/zero_count_5/ToFloat/Switch:0p
6linear/zero_fraction/total_zero/zero_count_5/pred_id:06linear/zero_fraction/total_zero/zero_count_5/pred_id:02?

?

Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_5/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_5/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_6/cond_text6linear/zero_fraction/total_zero/zero_count_6/pred_id:07linear/zero_fraction/total_zero/zero_count_6/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_6/Const:0
6linear/zero_fraction/total_zero/zero_count_6/pred_id:0
7linear/zero_fraction/total_zero/zero_count_6/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_6/pred_id:06linear/zero_fraction/total_zero/zero_count_6/pred_id:0
?0
8linear/zero_fraction/total_zero/zero_count_6/cond_text_16linear/zero_fraction/total_zero/zero_count_6/pred_id:07linear/zero_fraction/total_zero/zero_count_6/switch_f:0*?
3linear/linear_model/native_country/weights/part_0:0
(linear/zero_fraction/total_size/Size_6:0
=linear/zero_fraction/total_zero/zero_count_6/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_6/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_6/mul:0
6linear/zero_fraction/total_zero/zero_count_6/pred_id:0
7linear/zero_fraction/total_zero/zero_count_6/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_6/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_6/zero_fraction/fraction:0i
(linear/zero_fraction/total_size/Size_6:0=linear/zero_fraction/total_zero/zero_count_6/ToFloat/Switch:0p
6linear/zero_fraction/total_zero/zero_count_6/pred_id:06linear/zero_fraction/total_zero/zero_count_6/pred_id:0?
3linear/linear_model/native_country/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_6/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_6/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_7/cond_text6linear/zero_fraction/total_zero/zero_count_7/pred_id:07linear/zero_fraction/total_zero/zero_count_7/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_7/Const:0
6linear/zero_fraction/total_zero/zero_count_7/pred_id:0
7linear/zero_fraction/total_zero/zero_count_7/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_7/pred_id:06linear/zero_fraction/total_zero/zero_count_7/pred_id:0
?0
8linear/zero_fraction/total_zero/zero_count_7/cond_text_16linear/zero_fraction/total_zero/zero_count_7/pred_id:07linear/zero_fraction/total_zero/zero_count_7/switch_f:0*?
@linear/linear_model/native_country_X_occupation/weights/part_0:0
(linear/zero_fraction/total_size/Size_7:0
=linear/zero_fraction/total_zero/zero_count_7/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_7/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_7/mul:0
6linear/zero_fraction/total_zero/zero_count_7/pred_id:0
7linear/zero_fraction/total_zero/zero_count_7/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_7/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_7/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_7/pred_id:06linear/zero_fraction/total_zero/zero_count_7/pred_id:0i
(linear/zero_fraction/total_size/Size_7:0=linear/zero_fraction/total_zero/zero_count_7/ToFloat/Switch:0?
@linear/linear_model/native_country_X_occupation/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_7/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_7/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_8/cond_text6linear/zero_fraction/total_zero/zero_count_8/pred_id:07linear/zero_fraction/total_zero/zero_count_8/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_8/Const:0
6linear/zero_fraction/total_zero/zero_count_8/pred_id:0
7linear/zero_fraction/total_zero/zero_count_8/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_8/pred_id:06linear/zero_fraction/total_zero/zero_count_8/pred_id:0
?0
8linear/zero_fraction/total_zero/zero_count_8/cond_text_16linear/zero_fraction/total_zero/zero_count_8/pred_id:07linear/zero_fraction/total_zero/zero_count_8/switch_f:0*?
/linear/linear_model/occupation/weights/part_0:0
(linear/zero_fraction/total_size/Size_8:0
=linear/zero_fraction/total_zero/zero_count_8/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_8/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_8/mul:0
6linear/zero_fraction/total_zero/zero_count_8/pred_id:0
7linear/zero_fraction/total_zero/zero_count_8/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_8/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_8/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_8/zero_fraction/fraction:0p
6linear/zero_fraction/total_zero/zero_count_8/pred_id:06linear/zero_fraction/total_zero/zero_count_8/pred_id:0?
/linear/linear_model/occupation/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp/Switch:0i
(linear/zero_fraction/total_size/Size_8:0=linear/zero_fraction/total_zero/zero_count_8/ToFloat/Switch:02?

?

Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_8/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_8/zero_fraction/cond/pred_id:0
?
6linear/zero_fraction/total_zero/zero_count_9/cond_text6linear/zero_fraction/total_zero/zero_count_9/pred_id:07linear/zero_fraction/total_zero/zero_count_9/switch_t:0 *?
4linear/zero_fraction/total_zero/zero_count_9/Const:0
6linear/zero_fraction/total_zero/zero_count_9/pred_id:0
7linear/zero_fraction/total_zero/zero_count_9/switch_t:0p
6linear/zero_fraction/total_zero/zero_count_9/pred_id:06linear/zero_fraction/total_zero/zero_count_9/pred_id:0
?0
8linear/zero_fraction/total_zero/zero_count_9/cond_text_16linear/zero_fraction/total_zero/zero_count_9/pred_id:07linear/zero_fraction/total_zero/zero_count_9/switch_f:0*?
1linear/linear_model/relationship/weights/part_0:0
(linear/zero_fraction/total_size/Size_9:0
=linear/zero_fraction/total_zero/zero_count_9/ToFloat/Switch:0
6linear/zero_fraction/total_zero/zero_count_9/ToFloat:0
2linear/zero_fraction/total_zero/zero_count_9/mul:0
6linear/zero_fraction/total_zero/zero_count_9/pred_id:0
7linear/zero_fraction/total_zero/zero_count_9/switch_f:0
Hlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqual/y:0
Flinear/zero_fraction/total_zero/zero_count_9/zero_fraction/LessEqual:0
Rlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/ReadVariableOp/Switch:0
Klinear/zero_fraction/total_zero/zero_count_9/zero_fraction/ReadVariableOp:0
Alinear/zero_fraction/total_zero/zero_count_9/zero_fraction/Size:0
Flinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Cast:0
Glinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Merge:0
Glinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Merge:1
Hlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Switch:0
Hlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Switch:1
Tlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/zeros:0
Vlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_f:0
Jlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_t:0
Tlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/Cast_1:0
Slinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/sub:0
Wlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/counts_to_fraction/truediv:0
Elinear/zero_fraction/total_zero/zero_count_9/zero_fraction/fraction:0?
1linear/linear_model/relationship/weights/part_0:0Rlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/ReadVariableOp/Switch:0i
(linear/zero_fraction/total_size/Size_9:0=linear/zero_fraction/total_zero/zero_count_9/ToFloat/Switch:0p
6linear/zero_fraction/total_zero/zero_count_9/pred_id:06linear/zero_fraction/total_zero/zero_count_9/pred_id:02?

?

Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/cond_textIlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_t:0 *?	
Klinear/zero_fraction/total_zero/zero_count_9/zero_fraction/ReadVariableOp:0
Flinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/Cast:0
Tlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/Const:0
_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Xlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual:0
]linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/nonzero_count:0
Ulinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_t:0?
Klinear/zero_fraction/total_zero/zero_count_9/zero_fraction/ReadVariableOp:0_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:02?

?

Klinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/cond_text_1Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_f:0*?
Klinear/zero_fraction/total_zero/zero_count_9/zero_fraction/ReadVariableOp:0
Vlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/Const:0
alinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
Zlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual:0
_linear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Wlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/zeros:0
Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0
Jlinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/switch_f:0?
Klinear/zero_fraction/total_zero/zero_count_9/zero_fraction/ReadVariableOp:0alinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0Ilinear/zero_fraction/total_zero/zero_count_9/zero_fraction/cond/pred_id:0
?
7linear/zero_fraction/total_zero/zero_count_10/cond_text7linear/zero_fraction/total_zero/zero_count_10/pred_id:08linear/zero_fraction/total_zero/zero_count_10/switch_t:0 *?
5linear/zero_fraction/total_zero/zero_count_10/Const:0
7linear/zero_fraction/total_zero/zero_count_10/pred_id:0
8linear/zero_fraction/total_zero/zero_count_10/switch_t:0r
7linear/zero_fraction/total_zero/zero_count_10/pred_id:07linear/zero_fraction/total_zero/zero_count_10/pred_id:0
?0
9linear/zero_fraction/total_zero/zero_count_10/cond_text_17linear/zero_fraction/total_zero/zero_count_10/pred_id:08linear/zero_fraction/total_zero/zero_count_10/switch_f:0*?
.linear/linear_model/workclass/weights/part_0:0
)linear/zero_fraction/total_size/Size_10:0
>linear/zero_fraction/total_zero/zero_count_10/ToFloat/Switch:0
7linear/zero_fraction/total_zero/zero_count_10/ToFloat:0
3linear/zero_fraction/total_zero/zero_count_10/mul:0
7linear/zero_fraction/total_zero/zero_count_10/pred_id:0
8linear/zero_fraction/total_zero/zero_count_10/switch_f:0
Ilinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqual/y:0
Glinear/zero_fraction/total_zero/zero_count_10/zero_fraction/LessEqual:0
Slinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp/Switch:0
Llinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp:0
Blinear/zero_fraction/total_zero/zero_count_10/zero_fraction/Size:0
Glinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Cast:0
Hlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Merge:0
Hlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Merge:1
Ilinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Switch:0
Ilinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Switch:1
Ulinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/zeros:0
Wlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_f:0
Klinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_t:0
Ulinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/Cast:0
Wlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/Cast_1:0
Tlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/sub:0
Xlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/counts_to_fraction/truediv:0
Flinear/zero_fraction/total_zero/zero_count_10/zero_fraction/fraction:0?
.linear/linear_model/workclass/weights/part_0:0Slinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp/Switch:0r
7linear/zero_fraction/total_zero/zero_count_10/pred_id:07linear/zero_fraction/total_zero/zero_count_10/pred_id:0k
)linear/zero_fraction/total_size/Size_10:0>linear/zero_fraction/total_zero/zero_count_10/ToFloat/Switch:02?
?
Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/cond_textJlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_t:0 *?	
Llinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp:0
Glinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/Cast:0
Ulinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/Cast:0
Vlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/Const:0
`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual/Switch:1
Ylinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual:0
^linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/nonzero_count:0
Vlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_t:0?
Llinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp:0`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero/NotEqual/Switch:1?
Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:02?

?

Llinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/cond_text_1Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0Klinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_f:0*?
Llinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp:0
Wlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/Cast:0
Xlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/Const:0
blinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0
[linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual:0
`linear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/nonzero_count:0
Xlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/zeros:0
Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0
Klinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/switch_f:0?
Llinear/zero_fraction/total_zero/zero_count_10/zero_fraction/ReadVariableOp:0blinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/count_nonzero_1/NotEqual/Switch:0?
Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0Jlinear/zero_fraction/total_zero/zero_count_10/zero_fraction/cond/pred_id:0
?
%linear/zero_fraction_1/cond/cond_text%linear/zero_fraction_1/cond/pred_id:0&linear/zero_fraction_1/cond/switch_t:0 *?
<linear/linear_model/linear_model/linear_model/weighted_sum:0
"linear/zero_fraction_1/cond/Cast:0
0linear/zero_fraction_1/cond/count_nonzero/Cast:0
1linear/zero_fraction_1/cond/count_nonzero/Const:0
;linear/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
4linear/zero_fraction_1/cond/count_nonzero/NotEqual:0
9linear/zero_fraction_1/cond/count_nonzero/nonzero_count:0
1linear/zero_fraction_1/cond/count_nonzero/zeros:0
%linear/zero_fraction_1/cond/pred_id:0
&linear/zero_fraction_1/cond/switch_t:0N
%linear/zero_fraction_1/cond/pred_id:0%linear/zero_fraction_1/cond/pred_id:0{
<linear/linear_model/linear_model/linear_model/weighted_sum:0;linear/zero_fraction_1/cond/count_nonzero/NotEqual/Switch:1
?
'linear/zero_fraction_1/cond/cond_text_1%linear/zero_fraction_1/cond/pred_id:0&linear/zero_fraction_1/cond/switch_f:0*?
<linear/linear_model/linear_model/linear_model/weighted_sum:0
2linear/zero_fraction_1/cond/count_nonzero_1/Cast:0
3linear/zero_fraction_1/cond/count_nonzero_1/Const:0
=linear/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0
6linear/zero_fraction_1/cond/count_nonzero_1/NotEqual:0
;linear/zero_fraction_1/cond/count_nonzero_1/nonzero_count:0
3linear/zero_fraction_1/cond/count_nonzero_1/zeros:0
%linear/zero_fraction_1/cond/pred_id:0
&linear/zero_fraction_1/cond/switch_f:0N
%linear/zero_fraction_1/cond/pred_id:0%linear/zero_fraction_1/cond/pred_id:0}
<linear/linear_model/linear_model/linear_model/weighted_sum:0=linear/zero_fraction_1/cond/count_nonzero_1/NotEqual/Switch:0"m
global_step^\
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H"%
saved_model_main_op


group_deps"?
	summaries?
?
/dnn/dnn/hiddenlayer_0/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_0/activation:0
/dnn/dnn/hiddenlayer_1/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_1/activation:0
/dnn/dnn/hiddenlayer_2/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_2/activation:0
/dnn/dnn/hiddenlayer_3/fraction_of_zero_values:0
"dnn/dnn/hiddenlayer_3/activation:0
(dnn/dnn/logits/fraction_of_zero_values:0
dnn/dnn/logits/activation:0
linear/bias:0
!linear/fraction_of_zero_weights:0
'linear/linear/fraction_of_zero_values:0
linear/linear/activation:0"?8
trainable_variables?7?7
?
^dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0:0cdnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Assignrdnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Read/ReadVariableOp:0"c
Udnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weightsd  "d(2{dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
?
Zdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Assignndnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Read/ReadVariableOp:0"_
Qdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weightsd  "d(2wdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
?
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_0/kernelBd  "Bd(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/biasd "d(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kerneldF  "dF(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/biasF "F(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign5dnn/hiddenlayer_2/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_2/kernelF0  "F0(2<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign3dnn/hiddenlayer_2/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_2/bias0 "0(21dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_3/kernel/part_0:0&dnn/hiddenlayer_3/kernel/part_0/Assign5dnn/hiddenlayer_3/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_3/kernel0"  "0"(2<dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_3/bias/part_0:0$dnn/hiddenlayer_3/bias/part_0/Assign3dnn/hiddenlayer_3/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_3/bias" ""(21dnn/hiddenlayer_3/bias/part_0/Initializer/zeros:08
?
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel"  ""(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
?
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08
?
3linear/linear_model/age_bucketized/weights/part_0:08linear/linear_model/age_bucketized/weights/part_0/AssignGlinear/linear_model/age_bucketized/weights/part_0/Read/ReadVariableOp:0"8
*linear/linear_model/age_bucketized/weights  "(2Elinear/linear_model/age_bucketized/weights/part_0/Initializer/zeros:08
?
Glinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0:0Llinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/Assign[linear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/Read/ReadVariableOp:0"P
>linear/linear_model/age_bucketized_X_occupation_X_race/weights??=  "??=(2Ylinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/Initializer/zeros:08
?
.linear/linear_model/education/weights/part_0:03linear/linear_model/education/weights/part_0/AssignBlinear/linear_model/education/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/education/weights  "(2@linear/linear_model/education/weights/part_0/Initializer/zeros:08
?
;linear/linear_model/education_X_occupation/weights/part_0:0@linear/linear_model/education_X_occupation/weights/part_0/AssignOlinear/linear_model/education_X_occupation/weights/part_0/Read/ReadVariableOp:0"B
2linear/linear_model/education_X_occupation/weights?N  "?N(2Mlinear/linear_model/education_X_occupation/weights/part_0/Initializer/zeros:08
?
+linear/linear_model/gender/weights/part_0:00linear/linear_model/gender/weights/part_0/Assign?linear/linear_model/gender/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/gender/weights  "(2=linear/linear_model/gender/weights/part_0/Initializer/zeros:08
?
3linear/linear_model/marital_status/weights/part_0:08linear/linear_model/marital_status/weights/part_0/AssignGlinear/linear_model/marital_status/weights/part_0/Read/ReadVariableOp:0"8
*linear/linear_model/marital_status/weights  "(2Elinear/linear_model/marital_status/weights/part_0/Initializer/zeros:08
?
3linear/linear_model/native_country/weights/part_0:08linear/linear_model/native_country/weights/part_0/AssignGlinear/linear_model/native_country/weights/part_0/Read/ReadVariableOp:0"8
*linear/linear_model/native_country/weightsd  "d(2Elinear/linear_model/native_country/weights/part_0/Initializer/zeros:08
?
@linear/linear_model/native_country_X_occupation/weights/part_0:0Elinear/linear_model/native_country_X_occupation/weights/part_0/AssignTlinear/linear_model/native_country_X_occupation/weights/part_0/Read/ReadVariableOp:0"G
7linear/linear_model/native_country_X_occupation/weights?N  "?N(2Rlinear/linear_model/native_country_X_occupation/weights/part_0/Initializer/zeros:08
?
/linear/linear_model/occupation/weights/part_0:04linear/linear_model/occupation/weights/part_0/AssignClinear/linear_model/occupation/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/occupation/weightsd  "d(2Alinear/linear_model/occupation/weights/part_0/Initializer/zeros:08
?
1linear/linear_model/relationship/weights/part_0:06linear/linear_model/relationship/weights/part_0/AssignElinear/linear_model/relationship/weights/part_0/Read/ReadVariableOp:0"6
(linear/linear_model/relationship/weights  "(2Clinear/linear_model/relationship/weights/part_0/Initializer/zeros:08
?
.linear/linear_model/workclass/weights/part_0:03linear/linear_model/workclass/weights/part_0/AssignBlinear/linear_model/workclass/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/workclass/weights	  "	(2@linear/linear_model/workclass/weights/part_0/Initializer/zeros:08
?
)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08"?8
	variables?8?8
Z
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0H
?
^dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0:0cdnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Assignrdnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Read/ReadVariableOp:0"c
Udnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weightsd  "d(2{dnn/input_from_feature_columns/input_layer/native_country_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
?
Zdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0:0_dnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Assignndnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Read/ReadVariableOp:0"_
Qdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weightsd  "d(2wdnn/input_from_feature_columns/input_layer/occupation_embedding/embedding_weights/part_0/Initializer/truncated_normal:08
?
!dnn/hiddenlayer_0/kernel/part_0:0&dnn/hiddenlayer_0/kernel/part_0/Assign5dnn/hiddenlayer_0/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_0/kernelBd  "Bd(2<dnn/hiddenlayer_0/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_0/bias/part_0:0$dnn/hiddenlayer_0/bias/part_0/Assign3dnn/hiddenlayer_0/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_0/biasd "d(21dnn/hiddenlayer_0/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_1/kernel/part_0:0&dnn/hiddenlayer_1/kernel/part_0/Assign5dnn/hiddenlayer_1/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_1/kerneldF  "dF(2<dnn/hiddenlayer_1/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_1/bias/part_0:0$dnn/hiddenlayer_1/bias/part_0/Assign3dnn/hiddenlayer_1/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_1/biasF "F(21dnn/hiddenlayer_1/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_2/kernel/part_0:0&dnn/hiddenlayer_2/kernel/part_0/Assign5dnn/hiddenlayer_2/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_2/kernelF0  "F0(2<dnn/hiddenlayer_2/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_2/bias/part_0:0$dnn/hiddenlayer_2/bias/part_0/Assign3dnn/hiddenlayer_2/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_2/bias0 "0(21dnn/hiddenlayer_2/bias/part_0/Initializer/zeros:08
?
!dnn/hiddenlayer_3/kernel/part_0:0&dnn/hiddenlayer_3/kernel/part_0/Assign5dnn/hiddenlayer_3/kernel/part_0/Read/ReadVariableOp:0"&
dnn/hiddenlayer_3/kernel0"  "0"(2<dnn/hiddenlayer_3/kernel/part_0/Initializer/random_uniform:08
?
dnn/hiddenlayer_3/bias/part_0:0$dnn/hiddenlayer_3/bias/part_0/Assign3dnn/hiddenlayer_3/bias/part_0/Read/ReadVariableOp:0"!
dnn/hiddenlayer_3/bias" ""(21dnn/hiddenlayer_3/bias/part_0/Initializer/zeros:08
?
dnn/logits/kernel/part_0:0dnn/logits/kernel/part_0/Assign.dnn/logits/kernel/part_0/Read/ReadVariableOp:0"
dnn/logits/kernel"  ""(25dnn/logits/kernel/part_0/Initializer/random_uniform:08
?
dnn/logits/bias/part_0:0dnn/logits/bias/part_0/Assign,dnn/logits/bias/part_0/Read/ReadVariableOp:0"
dnn/logits/bias "(2*dnn/logits/bias/part_0/Initializer/zeros:08
?
3linear/linear_model/age_bucketized/weights/part_0:08linear/linear_model/age_bucketized/weights/part_0/AssignGlinear/linear_model/age_bucketized/weights/part_0/Read/ReadVariableOp:0"8
*linear/linear_model/age_bucketized/weights  "(2Elinear/linear_model/age_bucketized/weights/part_0/Initializer/zeros:08
?
Glinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0:0Llinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/Assign[linear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/Read/ReadVariableOp:0"P
>linear/linear_model/age_bucketized_X_occupation_X_race/weights??=  "??=(2Ylinear/linear_model/age_bucketized_X_occupation_X_race/weights/part_0/Initializer/zeros:08
?
.linear/linear_model/education/weights/part_0:03linear/linear_model/education/weights/part_0/AssignBlinear/linear_model/education/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/education/weights  "(2@linear/linear_model/education/weights/part_0/Initializer/zeros:08
?
;linear/linear_model/education_X_occupation/weights/part_0:0@linear/linear_model/education_X_occupation/weights/part_0/AssignOlinear/linear_model/education_X_occupation/weights/part_0/Read/ReadVariableOp:0"B
2linear/linear_model/education_X_occupation/weights?N  "?N(2Mlinear/linear_model/education_X_occupation/weights/part_0/Initializer/zeros:08
?
+linear/linear_model/gender/weights/part_0:00linear/linear_model/gender/weights/part_0/Assign?linear/linear_model/gender/weights/part_0/Read/ReadVariableOp:0"0
"linear/linear_model/gender/weights  "(2=linear/linear_model/gender/weights/part_0/Initializer/zeros:08
?
3linear/linear_model/marital_status/weights/part_0:08linear/linear_model/marital_status/weights/part_0/AssignGlinear/linear_model/marital_status/weights/part_0/Read/ReadVariableOp:0"8
*linear/linear_model/marital_status/weights  "(2Elinear/linear_model/marital_status/weights/part_0/Initializer/zeros:08
?
3linear/linear_model/native_country/weights/part_0:08linear/linear_model/native_country/weights/part_0/AssignGlinear/linear_model/native_country/weights/part_0/Read/ReadVariableOp:0"8
*linear/linear_model/native_country/weightsd  "d(2Elinear/linear_model/native_country/weights/part_0/Initializer/zeros:08
?
@linear/linear_model/native_country_X_occupation/weights/part_0:0Elinear/linear_model/native_country_X_occupation/weights/part_0/AssignTlinear/linear_model/native_country_X_occupation/weights/part_0/Read/ReadVariableOp:0"G
7linear/linear_model/native_country_X_occupation/weights?N  "?N(2Rlinear/linear_model/native_country_X_occupation/weights/part_0/Initializer/zeros:08
?
/linear/linear_model/occupation/weights/part_0:04linear/linear_model/occupation/weights/part_0/AssignClinear/linear_model/occupation/weights/part_0/Read/ReadVariableOp:0"4
&linear/linear_model/occupation/weightsd  "d(2Alinear/linear_model/occupation/weights/part_0/Initializer/zeros:08
?
1linear/linear_model/relationship/weights/part_0:06linear/linear_model/relationship/weights/part_0/AssignElinear/linear_model/relationship/weights/part_0/Read/ReadVariableOp:0"6
(linear/linear_model/relationship/weights  "(2Clinear/linear_model/relationship/weights/part_0/Initializer/zeros:08
?
.linear/linear_model/workclass/weights/part_0:03linear/linear_model/workclass/weights/part_0/AssignBlinear/linear_model/workclass/weights/part_0/Read/ReadVariableOp:0"3
%linear/linear_model/workclass/weights	  "	(2@linear/linear_model/workclass/weights/part_0/Initializer/zeros:08
?
)linear/linear_model/bias_weights/part_0:0.linear/linear_model/bias_weights/part_0/Assign=linear/linear_model/bias_weights/part_0/Read/ReadVariableOp:0"+
 linear/linear_model/bias_weights "(2;linear/linear_model/bias_weights/part_0/Initializer/zeros:08"?
table_initializer?
?
ydnn/input_from_feature_columns/input_layer/education_indicator/education_lookup/hash_table/table_init/LookupTableImportV2
sdnn/input_from_feature_columns/input_layer/gender_indicator/gender_lookup/hash_table/table_init/LookupTableImportV2
?dnn/input_from_feature_columns/input_layer/marital_status_indicator/marital_status_lookup/hash_table/table_init/LookupTableImportV2
odnn/input_from_feature_columns/input_layer/race_indicator/race_lookup/hash_table/table_init/LookupTableImportV2
dnn/input_from_feature_columns/input_layer/relationship_indicator/relationship_lookup/hash_table/table_init/LookupTableImportV2
ydnn/input_from_feature_columns/input_layer/workclass_indicator/workclass_lookup/hash_table/table_init/LookupTableImportV2
?linear/linear_model/linear_model/linear_model/age_bucketized_X_occupation_X_race/race_lookup/hash_table/table_init/LookupTableImportV2
rlinear/linear_model/linear_model/linear_model/education/education_lookup/hash_table/table_init/LookupTableImportV2
llinear/linear_model/linear_model/linear_model/gender/gender_lookup/hash_table/table_init/LookupTableImportV2
|linear/linear_model/linear_model/linear_model/marital_status/marital_status_lookup/hash_table/table_init/LookupTableImportV2
xlinear/linear_model/linear_model/linear_model/relationship/relationship_lookup/hash_table/table_init/LookupTableImportV2
rlinear/linear_model/linear_model/linear_model/workclass/workclass_lookup/hash_table/table_init/LookupTableImportV2*?
predict?
5
hours_per_week#
Placeholder_12:0?????????
*
gender 
Placeholder:0?????????
2
relationship"
Placeholder_4:0?????????
0

occupation"
Placeholder_6:0?????????
4
marital_status"
Placeholder_3:0?????????
*
race"
Placeholder_1:0?????????
4
native_country"
Placeholder_7:0?????????
3
education_num"
Placeholder_9:0?????????
/
	workclass"
Placeholder_5:0?????????
3
capital_gain#
Placeholder_10:0?????????
)
age"
Placeholder_8:0?????????
/
	education"
Placeholder_2:0?????????
3
capital_loss#
Placeholder_11:0?????????@
classes5
head/predictions/str_classes:0??????????
all_class_ids.
head/predictions/Tile:0?????????>
logistic2
head/predictions/logistic:0??????????
all_classes0
head/predictions/Tile_1:0?????????H
probabilities7
 head/predictions/probabilities:0?????????&
logits
add:0?????????A
	class_ids4
head/predictions/ExpandDims:0	?????????tensorflow/serving/predict