
Ν’
^
AssignVariableOp
resource
value"dtype"
dtypetype"
validate_shapebool( 
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(

NoOp
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
@
ReadVariableOp
resource
value"dtype"
dtypetype
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
Α
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ¨
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.8.22v2.8.1-10-g2ea19cbb5758ΘΈ
~
dense_1047/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*"
shared_namedense_1047/kernel
w
%dense_1047/kernel/Read/ReadVariableOpReadVariableOpdense_1047/kernel*
_output_shapes

:<<*
dtype0
v
dense_1047/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<* 
shared_namedense_1047/bias
o
#dense_1047/bias/Read/ReadVariableOpReadVariableOpdense_1047/bias*
_output_shapes
:<*
dtype0
~
dense_1048/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*"
shared_namedense_1048/kernel
w
%dense_1048/kernel/Read/ReadVariableOpReadVariableOpdense_1048/kernel*
_output_shapes

:<<*
dtype0
v
dense_1048/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<* 
shared_namedense_1048/bias
o
#dense_1048/bias/Read/ReadVariableOpReadVariableOpdense_1048/bias*
_output_shapes
:<*
dtype0
~
dense_1049/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*"
shared_namedense_1049/kernel
w
%dense_1049/kernel/Read/ReadVariableOpReadVariableOpdense_1049/kernel*
_output_shapes

:<2*
dtype0
v
dense_1049/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2* 
shared_namedense_1049/bias
o
#dense_1049/bias/Read/ReadVariableOpReadVariableOpdense_1049/bias*
_output_shapes
:2*
dtype0
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0

Adam/dense_1047/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*)
shared_nameAdam/dense_1047/kernel/m

,Adam/dense_1047/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1047/kernel/m*
_output_shapes

:<<*
dtype0

Adam/dense_1047/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameAdam/dense_1047/bias/m
}
*Adam/dense_1047/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1047/bias/m*
_output_shapes
:<*
dtype0

Adam/dense_1048/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*)
shared_nameAdam/dense_1048/kernel/m

,Adam/dense_1048/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1048/kernel/m*
_output_shapes

:<<*
dtype0

Adam/dense_1048/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameAdam/dense_1048/bias/m
}
*Adam/dense_1048/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1048/bias/m*
_output_shapes
:<*
dtype0

Adam/dense_1049/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*)
shared_nameAdam/dense_1049/kernel/m

,Adam/dense_1049/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1049/kernel/m*
_output_shapes

:<2*
dtype0

Adam/dense_1049/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameAdam/dense_1049/bias/m
}
*Adam/dense_1049/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1049/bias/m*
_output_shapes
:2*
dtype0

Adam/dense_1047/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*)
shared_nameAdam/dense_1047/kernel/v

,Adam/dense_1047/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1047/kernel/v*
_output_shapes

:<<*
dtype0

Adam/dense_1047/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameAdam/dense_1047/bias/v
}
*Adam/dense_1047/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1047/bias/v*
_output_shapes
:<*
dtype0

Adam/dense_1048/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*)
shared_nameAdam/dense_1048/kernel/v

,Adam/dense_1048/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1048/kernel/v*
_output_shapes

:<<*
dtype0

Adam/dense_1048/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameAdam/dense_1048/bias/v
}
*Adam/dense_1048/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1048/bias/v*
_output_shapes
:<*
dtype0

Adam/dense_1049/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*)
shared_nameAdam/dense_1049/kernel/v

,Adam/dense_1049/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1049/kernel/v*
_output_shapes

:<2*
dtype0

Adam/dense_1049/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameAdam/dense_1049/bias/v
}
*Adam/dense_1049/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1049/bias/v*
_output_shapes
:2*
dtype0

NoOpNoOp
+
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*Ε*
value»*BΈ* B±*
Α
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses*
¦

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses*
°
%iter

&beta_1

'beta_2
	(decay
)learning_ratemDmEmFmGmHmIvJvKvLvMvNvO*
.
0
1
2
3
4
5*
.
0
1
2
3
4
5*
* 
°
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses*
* 
* 
* 

/serving_default* 
a[
VARIABLE_VALUEdense_1047/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_1047/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
a[
VARIABLE_VALUEdense_1048/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_1048/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses*
* 
* 
a[
VARIABLE_VALUEdense_1049/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_1049/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*

0
1*

0
1*
* 

:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses*
* 
* 
LF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE*
PJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE*
NH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE*
^X
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE*
* 

0
1
2*

?0*
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
* 
8
	@total
	Acount
B	variables
C	keras_api*
SM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE*
SM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE*

@0
A1*

B	variables*
~
VARIABLE_VALUEAdam/dense_1047/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1047/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1048/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1048/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1049/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1049/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1047/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1047/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1048/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1048/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1049/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1049/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
serving_default_input_350Placeholder*'
_output_shapes
:?????????<*
dtype0*
shape:?????????<
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_350dense_1047/kerneldense_1047/biasdense_1048/kerneldense_1048/biasdense_1049/kerneldense_1049/bias*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 */
f*R(
&__inference_signature_wrapper_79805307
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 


StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%dense_1047/kernel/Read/ReadVariableOp#dense_1047/bias/Read/ReadVariableOp%dense_1048/kernel/Read/ReadVariableOp#dense_1048/bias/Read/ReadVariableOp%dense_1049/kernel/Read/ReadVariableOp#dense_1049/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/dense_1047/kernel/m/Read/ReadVariableOp*Adam/dense_1047/bias/m/Read/ReadVariableOp,Adam/dense_1048/kernel/m/Read/ReadVariableOp*Adam/dense_1048/bias/m/Read/ReadVariableOp,Adam/dense_1049/kernel/m/Read/ReadVariableOp*Adam/dense_1049/bias/m/Read/ReadVariableOp,Adam/dense_1047/kernel/v/Read/ReadVariableOp*Adam/dense_1047/bias/v/Read/ReadVariableOp,Adam/dense_1048/kernel/v/Read/ReadVariableOp*Adam/dense_1048/bias/v/Read/ReadVariableOp,Adam/dense_1049/kernel/v/Read/ReadVariableOp*Adam/dense_1049/bias/v/Read/ReadVariableOpConst*&
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__traced_save_79805464

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1047/kerneldense_1047/biasdense_1048/kerneldense_1048/biasdense_1049/kerneldense_1049/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_1047/kernel/mAdam/dense_1047/bias/mAdam/dense_1048/kernel/mAdam/dense_1048/bias/mAdam/dense_1049/kernel/mAdam/dense_1049/bias/mAdam/dense_1047/kernel/vAdam/dense_1047/bias/vAdam/dense_1048/kernel/vAdam/dense_1048/bias/vAdam/dense_1049/kernel/vAdam/dense_1049/bias/v*%
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference__traced_restore_79805549Θ
Γ
₯
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805288

inputs;
)dense_1047_matmul_readvariableop_resource:<<8
*dense_1047_biasadd_readvariableop_resource:<;
)dense_1048_matmul_readvariableop_resource:<<8
*dense_1048_biasadd_readvariableop_resource:<;
)dense_1049_matmul_readvariableop_resource:<28
*dense_1049_biasadd_readvariableop_resource:2
identity’!dense_1047/BiasAdd/ReadVariableOp’ dense_1047/MatMul/ReadVariableOp’!dense_1048/BiasAdd/ReadVariableOp’ dense_1048/MatMul/ReadVariableOp’!dense_1049/BiasAdd/ReadVariableOp’ dense_1049/MatMul/ReadVariableOp
 dense_1047/MatMul/ReadVariableOpReadVariableOp)dense_1047_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype0
dense_1047/MatMulMatMulinputs(dense_1047/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<
!dense_1047/BiasAdd/ReadVariableOpReadVariableOp*dense_1047_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0
dense_1047/BiasAddBiasAdddense_1047/MatMul:product:0)dense_1047/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<f
dense_1047/ReluReludense_1047/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<
 dense_1048/MatMul/ReadVariableOpReadVariableOp)dense_1048_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype0
dense_1048/MatMulMatMuldense_1047/Relu:activations:0(dense_1048/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<
!dense_1048/BiasAdd/ReadVariableOpReadVariableOp*dense_1048_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0
dense_1048/BiasAddBiasAdddense_1048/MatMul:product:0)dense_1048/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<f
dense_1048/ReluReludense_1048/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<
 dense_1049/MatMul/ReadVariableOpReadVariableOp)dense_1049_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype0
dense_1049/MatMulMatMuldense_1048/Relu:activations:0(dense_1049/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
!dense_1049/BiasAdd/ReadVariableOpReadVariableOp*dense_1049_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_1049/BiasAddBiasAdddense_1049/MatMul:product:0)dense_1049/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2j
IdentityIdentitydense_1049/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2
NoOpNoOp"^dense_1047/BiasAdd/ReadVariableOp!^dense_1047/MatMul/ReadVariableOp"^dense_1048/BiasAdd/ReadVariableOp!^dense_1048/MatMul/ReadVariableOp"^dense_1049/BiasAdd/ReadVariableOp!^dense_1049/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 2F
!dense_1047/BiasAdd/ReadVariableOp!dense_1047/BiasAdd/ReadVariableOp2D
 dense_1047/MatMul/ReadVariableOp dense_1047/MatMul/ReadVariableOp2F
!dense_1048/BiasAdd/ReadVariableOp!dense_1048/BiasAdd/ReadVariableOp2D
 dense_1048/MatMul/ReadVariableOp dense_1048/MatMul/ReadVariableOp2F
!dense_1049/BiasAdd/ReadVariableOp!dense_1049/BiasAdd/ReadVariableOp2D
 dense_1049/MatMul/ReadVariableOp dense_1049/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
Ϊ9
ΰ

!__inference__traced_save_79805464
file_prefix0
,savev2_dense_1047_kernel_read_readvariableop.
*savev2_dense_1047_bias_read_readvariableop0
,savev2_dense_1048_kernel_read_readvariableop.
*savev2_dense_1048_bias_read_readvariableop0
,savev2_dense_1049_kernel_read_readvariableop.
*savev2_dense_1049_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_dense_1047_kernel_m_read_readvariableop5
1savev2_adam_dense_1047_bias_m_read_readvariableop7
3savev2_adam_dense_1048_kernel_m_read_readvariableop5
1savev2_adam_dense_1048_bias_m_read_readvariableop7
3savev2_adam_dense_1049_kernel_m_read_readvariableop5
1savev2_adam_dense_1049_bias_m_read_readvariableop7
3savev2_adam_dense_1047_kernel_v_read_readvariableop5
1savev2_adam_dense_1047_bias_v_read_readvariableop7
3savev2_adam_dense_1048_kernel_v_read_readvariableop5
1savev2_adam_dense_1048_bias_v_read_readvariableop7
3savev2_adam_dense_1049_kernel_v_read_readvariableop5
1savev2_adam_dense_1049_bias_v_read_readvariableop
savev2_const

identity_1’MergeV2Checkpointsw
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*Z
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.parta
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: f

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: L

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :f
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*²
value¨B₯B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH‘
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B Χ

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_dense_1047_kernel_read_readvariableop*savev2_dense_1047_bias_read_readvariableop,savev2_dense_1048_kernel_read_readvariableop*savev2_dense_1048_bias_read_readvariableop,savev2_dense_1049_kernel_read_readvariableop*savev2_dense_1049_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_dense_1047_kernel_m_read_readvariableop1savev2_adam_dense_1047_bias_m_read_readvariableop3savev2_adam_dense_1048_kernel_m_read_readvariableop1savev2_adam_dense_1048_bias_m_read_readvariableop3savev2_adam_dense_1049_kernel_m_read_readvariableop1savev2_adam_dense_1049_bias_m_read_readvariableop3savev2_adam_dense_1047_kernel_v_read_readvariableop1savev2_adam_dense_1047_bias_v_read_readvariableop3savev2_adam_dense_1048_kernel_v_read_readvariableop1savev2_adam_dense_1048_bias_v_read_readvariableop3savev2_adam_dense_1049_kernel_v_read_readvariableop1savev2_adam_dense_1049_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *(
dtypes
2	
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 f
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: Q

Identity_1IdentityIdentity:output:0^NoOp*
T0*
_output_shapes
: [
NoOpNoOp^MergeV2Checkpoints*"
_acd_function_control_output(*
_output_shapes
 "!

identity_1Identity_1:output:0*·
_input_shapes₯
’: :<<:<:<<:<:<2:2: : : : : : : :<<:<:<<:<:<2:2:<<:<:<<:<:<2:2: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:<<: 

_output_shapes
:<:$ 

_output_shapes

:<<: 

_output_shapes
:<:$ 

_output_shapes

:<2: 

_output_shapes
:2:

_output_shapes
: :

_output_shapes
: :	

_output_shapes
: :


_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :$ 

_output_shapes

:<<: 

_output_shapes
:<:$ 

_output_shapes

:<<: 

_output_shapes
:<:$ 

_output_shapes

:<2: 

_output_shapes
:2:$ 

_output_shapes

:<<: 

_output_shapes
:<:$ 

_output_shapes

:<<: 

_output_shapes
:<:$ 

_output_shapes

:<2: 

_output_shapes
:2:

_output_shapes
: 
ρ
»
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805200
	input_350%
dense_1047_79805184:<<!
dense_1047_79805186:<%
dense_1048_79805189:<<!
dense_1048_79805191:<%
dense_1049_79805194:<2!
dense_1049_79805196:2
identity’"dense_1047/StatefulPartitionedCall’"dense_1048/StatefulPartitionedCall’"dense_1049/StatefulPartitionedCall
"dense_1047/StatefulPartitionedCallStatefulPartitionedCall	input_350dense_1047_79805184dense_1047_79805186*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1047_layer_call_and_return_conditional_losses_79805007£
"dense_1048/StatefulPartitionedCallStatefulPartitionedCall+dense_1047/StatefulPartitionedCall:output:0dense_1048_79805189dense_1048_79805191*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1048_layer_call_and_return_conditional_losses_79805024£
"dense_1049/StatefulPartitionedCallStatefulPartitionedCall+dense_1048/StatefulPartitionedCall:output:0dense_1049_79805194dense_1049_79805196*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1049_layer_call_and_return_conditional_losses_79805040z
IdentityIdentity+dense_1049/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2΅
NoOpNoOp#^dense_1047/StatefulPartitionedCall#^dense_1048/StatefulPartitionedCall#^dense_1049/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 2H
"dense_1047/StatefulPartitionedCall"dense_1047/StatefulPartitionedCall2H
"dense_1048/StatefulPartitionedCall"dense_1048/StatefulPartitionedCall2H
"dense_1049/StatefulPartitionedCall"dense_1049/StatefulPartitionedCall:R N
'
_output_shapes
:?????????<
#
_user_specified_name	input_350
Κ

-__inference_dense_1049_layer_call_fn_79805356

inputs
unknown:<2
	unknown_0:2
identity’StatefulPartitionedCallέ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1049_layer_call_and_return_conditional_losses_79805040o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????<: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
Λ	
ω
H__inference_dense_1049_layer_call_and_return_conditional_losses_79805366

inputs0
matmul_readvariableop_resource:<2-
biasadd_readvariableop_resource:2
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
θ
Έ
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805047

inputs%
dense_1047_79805008:<<!
dense_1047_79805010:<%
dense_1048_79805025:<<!
dense_1048_79805027:<%
dense_1049_79805041:<2!
dense_1049_79805043:2
identity’"dense_1047/StatefulPartitionedCall’"dense_1048/StatefulPartitionedCall’"dense_1049/StatefulPartitionedCallώ
"dense_1047/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1047_79805008dense_1047_79805010*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1047_layer_call_and_return_conditional_losses_79805007£
"dense_1048/StatefulPartitionedCallStatefulPartitionedCall+dense_1047/StatefulPartitionedCall:output:0dense_1048_79805025dense_1048_79805027*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1048_layer_call_and_return_conditional_losses_79805024£
"dense_1049/StatefulPartitionedCallStatefulPartitionedCall+dense_1048/StatefulPartitionedCall:output:0dense_1049_79805041dense_1049_79805043*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1049_layer_call_and_return_conditional_losses_79805040z
IdentityIdentity+dense_1049/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2΅
NoOpNoOp#^dense_1047/StatefulPartitionedCall#^dense_1048/StatefulPartitionedCall#^dense_1049/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 2H
"dense_1047/StatefulPartitionedCall"dense_1047/StatefulPartitionedCall2H
"dense_1048/StatefulPartitionedCall"dense_1048/StatefulPartitionedCall2H
"dense_1049/StatefulPartitionedCall"dense_1049/StatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
ζ#
³
#__inference__wrapped_model_79804989
	input_350J
8sequential_349_dense_1047_matmul_readvariableop_resource:<<G
9sequential_349_dense_1047_biasadd_readvariableop_resource:<J
8sequential_349_dense_1048_matmul_readvariableop_resource:<<G
9sequential_349_dense_1048_biasadd_readvariableop_resource:<J
8sequential_349_dense_1049_matmul_readvariableop_resource:<2G
9sequential_349_dense_1049_biasadd_readvariableop_resource:2
identity’0sequential_349/dense_1047/BiasAdd/ReadVariableOp’/sequential_349/dense_1047/MatMul/ReadVariableOp’0sequential_349/dense_1048/BiasAdd/ReadVariableOp’/sequential_349/dense_1048/MatMul/ReadVariableOp’0sequential_349/dense_1049/BiasAdd/ReadVariableOp’/sequential_349/dense_1049/MatMul/ReadVariableOp¨
/sequential_349/dense_1047/MatMul/ReadVariableOpReadVariableOp8sequential_349_dense_1047_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype0 
 sequential_349/dense_1047/MatMulMatMul	input_3507sequential_349/dense_1047/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<¦
0sequential_349/dense_1047/BiasAdd/ReadVariableOpReadVariableOp9sequential_349_dense_1047_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0Δ
!sequential_349/dense_1047/BiasAddBiasAdd*sequential_349/dense_1047/MatMul:product:08sequential_349/dense_1047/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<
sequential_349/dense_1047/ReluRelu*sequential_349/dense_1047/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<¨
/sequential_349/dense_1048/MatMul/ReadVariableOpReadVariableOp8sequential_349_dense_1048_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype0Γ
 sequential_349/dense_1048/MatMulMatMul,sequential_349/dense_1047/Relu:activations:07sequential_349/dense_1048/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<¦
0sequential_349/dense_1048/BiasAdd/ReadVariableOpReadVariableOp9sequential_349_dense_1048_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0Δ
!sequential_349/dense_1048/BiasAddBiasAdd*sequential_349/dense_1048/MatMul:product:08sequential_349/dense_1048/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<
sequential_349/dense_1048/ReluRelu*sequential_349/dense_1048/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<¨
/sequential_349/dense_1049/MatMul/ReadVariableOpReadVariableOp8sequential_349_dense_1049_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype0Γ
 sequential_349/dense_1049/MatMulMatMul,sequential_349/dense_1048/Relu:activations:07sequential_349/dense_1049/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2¦
0sequential_349/dense_1049/BiasAdd/ReadVariableOpReadVariableOp9sequential_349_dense_1049_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Δ
!sequential_349/dense_1049/BiasAddBiasAdd*sequential_349/dense_1049/MatMul:product:08sequential_349/dense_1049/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2y
IdentityIdentity*sequential_349/dense_1049/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2υ
NoOpNoOp1^sequential_349/dense_1047/BiasAdd/ReadVariableOp0^sequential_349/dense_1047/MatMul/ReadVariableOp1^sequential_349/dense_1048/BiasAdd/ReadVariableOp0^sequential_349/dense_1048/MatMul/ReadVariableOp1^sequential_349/dense_1049/BiasAdd/ReadVariableOp0^sequential_349/dense_1049/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 2d
0sequential_349/dense_1047/BiasAdd/ReadVariableOp0sequential_349/dense_1047/BiasAdd/ReadVariableOp2b
/sequential_349/dense_1047/MatMul/ReadVariableOp/sequential_349/dense_1047/MatMul/ReadVariableOp2d
0sequential_349/dense_1048/BiasAdd/ReadVariableOp0sequential_349/dense_1048/BiasAdd/ReadVariableOp2b
/sequential_349/dense_1048/MatMul/ReadVariableOp/sequential_349/dense_1048/MatMul/ReadVariableOp2d
0sequential_349/dense_1049/BiasAdd/ReadVariableOp0sequential_349/dense_1049/BiasAdd/ReadVariableOp2b
/sequential_349/dense_1049/MatMul/ReadVariableOp/sequential_349/dense_1049/MatMul/ReadVariableOp:R N
'
_output_shapes
:?????????<
#
_user_specified_name	input_350
Ο

&__inference_signature_wrapper_79805307
	input_350
unknown:<<
	unknown_0:<
	unknown_1:<<
	unknown_2:<
	unknown_3:<2
	unknown_4:2
identity’StatefulPartitionedCallο
StatefulPartitionedCallStatefulPartitionedCall	input_350unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *,
f'R%
#__inference__wrapped_model_79804989o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????<
#
_user_specified_name	input_350
ϊ

1__inference_sequential_349_layer_call_fn_79805223

inputs
unknown:<<
	unknown_0:<
	unknown_1:<<
	unknown_2:<
	unknown_3:<2
	unknown_4:2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805047o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs


ω
H__inference_dense_1047_layer_call_and_return_conditional_losses_79805327

inputs0
matmul_readvariableop_resource:<<-
biasadd_readvariableop_resource:<
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????<w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
Κ

-__inference_dense_1047_layer_call_fn_79805316

inputs
unknown:<<
	unknown_0:<
identity’StatefulPartitionedCallέ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1047_layer_call_and_return_conditional_losses_79805007o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????<: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs


ω
H__inference_dense_1047_layer_call_and_return_conditional_losses_79805007

inputs0
matmul_readvariableop_resource:<<-
biasadd_readvariableop_resource:<
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????<w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
Κ

-__inference_dense_1048_layer_call_fn_79805336

inputs
unknown:<<
	unknown_0:<
identity’StatefulPartitionedCallέ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1048_layer_call_and_return_conditional_losses_79805024o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????<`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????<: : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
ρ
»
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805181
	input_350%
dense_1047_79805165:<<!
dense_1047_79805167:<%
dense_1048_79805170:<<!
dense_1048_79805172:<%
dense_1049_79805175:<2!
dense_1049_79805177:2
identity’"dense_1047/StatefulPartitionedCall’"dense_1048/StatefulPartitionedCall’"dense_1049/StatefulPartitionedCall
"dense_1047/StatefulPartitionedCallStatefulPartitionedCall	input_350dense_1047_79805165dense_1047_79805167*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1047_layer_call_and_return_conditional_losses_79805007£
"dense_1048/StatefulPartitionedCallStatefulPartitionedCall+dense_1047/StatefulPartitionedCall:output:0dense_1048_79805170dense_1048_79805172*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1048_layer_call_and_return_conditional_losses_79805024£
"dense_1049/StatefulPartitionedCallStatefulPartitionedCall+dense_1048/StatefulPartitionedCall:output:0dense_1049_79805175dense_1049_79805177*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1049_layer_call_and_return_conditional_losses_79805040z
IdentityIdentity+dense_1049/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2΅
NoOpNoOp#^dense_1047/StatefulPartitionedCall#^dense_1048/StatefulPartitionedCall#^dense_1049/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 2H
"dense_1047/StatefulPartitionedCall"dense_1047/StatefulPartitionedCall2H
"dense_1048/StatefulPartitionedCall"dense_1048/StatefulPartitionedCall2H
"dense_1049/StatefulPartitionedCall"dense_1049/StatefulPartitionedCall:R N
'
_output_shapes
:?????????<
#
_user_specified_name	input_350
f

$__inference__traced_restore_79805549
file_prefix4
"assignvariableop_dense_1047_kernel:<<0
"assignvariableop_1_dense_1047_bias:<6
$assignvariableop_2_dense_1048_kernel:<<0
"assignvariableop_3_dense_1048_bias:<6
$assignvariableop_4_dense_1049_kernel:<20
"assignvariableop_5_dense_1049_bias:2&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: >
,assignvariableop_13_adam_dense_1047_kernel_m:<<8
*assignvariableop_14_adam_dense_1047_bias_m:<>
,assignvariableop_15_adam_dense_1048_kernel_m:<<8
*assignvariableop_16_adam_dense_1048_bias_m:<>
,assignvariableop_17_adam_dense_1049_kernel_m:<28
*assignvariableop_18_adam_dense_1049_bias_m:2>
,assignvariableop_19_adam_dense_1047_kernel_v:<<8
*assignvariableop_20_adam_dense_1047_bias_v:<>
,assignvariableop_21_adam_dense_1048_kernel_v:<<8
*assignvariableop_22_adam_dense_1048_bias_v:<>
,assignvariableop_23_adam_dense_1049_kernel_v:<28
*assignvariableop_24_adam_dense_1049_bias_v:2
identity_26’AssignVariableOp’AssignVariableOp_1’AssignVariableOp_10’AssignVariableOp_11’AssignVariableOp_12’AssignVariableOp_13’AssignVariableOp_14’AssignVariableOp_15’AssignVariableOp_16’AssignVariableOp_17’AssignVariableOp_18’AssignVariableOp_19’AssignVariableOp_2’AssignVariableOp_20’AssignVariableOp_21’AssignVariableOp_22’AssignVariableOp_23’AssignVariableOp_24’AssignVariableOp_3’AssignVariableOp_4’AssignVariableOp_5’AssignVariableOp_6’AssignVariableOp_7’AssignVariableOp_8’AssignVariableOp_9
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*²
value¨B₯B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH€
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*G
value>B<B B B B B B B B B B B B B B B B B B B B B B B B B B  
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*|
_output_shapesj
h::::::::::::::::::::::::::*(
dtypes
2	[
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOpAssignVariableOp"assignvariableop_dense_1047_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_1047_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_1048_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_1048_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense_1049_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_1049_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0	*
_output_shapes
:
AssignVariableOp_6AssignVariableOpassignvariableop_6_adam_iterIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	]

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_7AssignVariableOpassignvariableop_7_adam_beta_1Identity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_8AssignVariableOpassignvariableop_8_adam_beta_2Identity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_9AssignVariableOpassignvariableop_9_adam_decayIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_10AssignVariableOp&assignvariableop_10_adam_learning_rateIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_11AssignVariableOpassignvariableop_11_totalIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_12AssignVariableOpassignvariableop_12_countIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_13AssignVariableOp,assignvariableop_13_adam_dense_1047_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_dense_1047_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_dense_1048_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_1048_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_dense_1049_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_1049_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_dense_1047_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_1047_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_dense_1048_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_1048_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_dense_1049_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_1049_bias_vIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype01
NoOpNoOp"/device:CPU:0*
_output_shapes
 υ
Identity_25Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: W
Identity_26IdentityIdentity_25:output:0^NoOp_1*
T0*
_output_shapes
: β
NoOp_1NoOp^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*"
_acd_function_control_output(*
_output_shapes
 "#
identity_26Identity_26:output:0*G
_input_shapes6
4: : : : : : : : : : : : : : : : : : : : : : : : : : 2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
	

1__inference_sequential_349_layer_call_fn_79805162
	input_350
unknown:<<
	unknown_0:<
	unknown_1:<<
	unknown_2:<
	unknown_3:<2
	unknown_4:2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	input_350unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805130o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????<
#
_user_specified_name	input_350
	

1__inference_sequential_349_layer_call_fn_79805062
	input_350
unknown:<<
	unknown_0:<
	unknown_1:<<
	unknown_2:<
	unknown_3:<2
	unknown_4:2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	input_350unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805047o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:R N
'
_output_shapes
:?????????<
#
_user_specified_name	input_350


ω
H__inference_dense_1048_layer_call_and_return_conditional_losses_79805024

inputs0
matmul_readvariableop_resource:<<-
biasadd_readvariableop_resource:<
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????<w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
ϊ

1__inference_sequential_349_layer_call_fn_79805240

inputs
unknown:<<
	unknown_0:<
	unknown_1:<<
	unknown_2:<
	unknown_3:<2
	unknown_4:2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
Tin
	2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*(
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8 *U
fPRN
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805130o
IdentityIdentity StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2`
NoOpNoOp^StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
θ
Έ
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805130

inputs%
dense_1047_79805114:<<!
dense_1047_79805116:<%
dense_1048_79805119:<<!
dense_1048_79805121:<%
dense_1049_79805124:<2!
dense_1049_79805126:2
identity’"dense_1047/StatefulPartitionedCall’"dense_1048/StatefulPartitionedCall’"dense_1049/StatefulPartitionedCallώ
"dense_1047/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1047_79805114dense_1047_79805116*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1047_layer_call_and_return_conditional_losses_79805007£
"dense_1048/StatefulPartitionedCallStatefulPartitionedCall+dense_1047/StatefulPartitionedCall:output:0dense_1048_79805119dense_1048_79805121*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????<*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1048_layer_call_and_return_conditional_losses_79805024£
"dense_1049/StatefulPartitionedCallStatefulPartitionedCall+dense_1048/StatefulPartitionedCall:output:0dense_1049_79805124dense_1049_79805126*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:?????????2*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_dense_1049_layer_call_and_return_conditional_losses_79805040z
IdentityIdentity+dense_1049/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2΅
NoOpNoOp#^dense_1047/StatefulPartitionedCall#^dense_1048/StatefulPartitionedCall#^dense_1049/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 2H
"dense_1047/StatefulPartitionedCall"dense_1047/StatefulPartitionedCall2H
"dense_1048/StatefulPartitionedCall"dense_1048/StatefulPartitionedCall2H
"dense_1049/StatefulPartitionedCall"dense_1049/StatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs


ω
H__inference_dense_1048_layer_call_and_return_conditional_losses_79805347

inputs0
matmul_readvariableop_resource:<<-
biasadd_readvariableop_resource:<
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<<*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:<*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<P
ReluReluBiasAdd:output:0*
T0*'
_output_shapes
:?????????<a
IdentityIdentityRelu:activations:0^NoOp*
T0*'
_output_shapes
:?????????<w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
Γ
₯
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805264

inputs;
)dense_1047_matmul_readvariableop_resource:<<8
*dense_1047_biasadd_readvariableop_resource:<;
)dense_1048_matmul_readvariableop_resource:<<8
*dense_1048_biasadd_readvariableop_resource:<;
)dense_1049_matmul_readvariableop_resource:<28
*dense_1049_biasadd_readvariableop_resource:2
identity’!dense_1047/BiasAdd/ReadVariableOp’ dense_1047/MatMul/ReadVariableOp’!dense_1048/BiasAdd/ReadVariableOp’ dense_1048/MatMul/ReadVariableOp’!dense_1049/BiasAdd/ReadVariableOp’ dense_1049/MatMul/ReadVariableOp
 dense_1047/MatMul/ReadVariableOpReadVariableOp)dense_1047_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype0
dense_1047/MatMulMatMulinputs(dense_1047/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<
!dense_1047/BiasAdd/ReadVariableOpReadVariableOp*dense_1047_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0
dense_1047/BiasAddBiasAdddense_1047/MatMul:product:0)dense_1047/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<f
dense_1047/ReluReludense_1047/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<
 dense_1048/MatMul/ReadVariableOpReadVariableOp)dense_1048_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype0
dense_1048/MatMulMatMuldense_1047/Relu:activations:0(dense_1048/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<
!dense_1048/BiasAdd/ReadVariableOpReadVariableOp*dense_1048_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0
dense_1048/BiasAddBiasAdddense_1048/MatMul:product:0)dense_1048/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<f
dense_1048/ReluReludense_1048/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<
 dense_1049/MatMul/ReadVariableOpReadVariableOp)dense_1049_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype0
dense_1049/MatMulMatMuldense_1048/Relu:activations:0(dense_1049/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
!dense_1049/BiasAdd/ReadVariableOpReadVariableOp*dense_1049_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_1049/BiasAddBiasAdddense_1049/MatMul:product:0)dense_1049/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2j
IdentityIdentitydense_1049/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2
NoOpNoOp"^dense_1047/BiasAdd/ReadVariableOp!^dense_1047/MatMul/ReadVariableOp"^dense_1048/BiasAdd/ReadVariableOp!^dense_1048/MatMul/ReadVariableOp"^dense_1049/BiasAdd/ReadVariableOp!^dense_1049/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 2F
!dense_1047/BiasAdd/ReadVariableOp!dense_1047/BiasAdd/ReadVariableOp2D
 dense_1047/MatMul/ReadVariableOp dense_1047/MatMul/ReadVariableOp2F
!dense_1048/BiasAdd/ReadVariableOp!dense_1048/BiasAdd/ReadVariableOp2D
 dense_1048/MatMul/ReadVariableOp dense_1048/MatMul/ReadVariableOp2F
!dense_1049/BiasAdd/ReadVariableOp!dense_1049/BiasAdd/ReadVariableOp2D
 dense_1049/MatMul/ReadVariableOp dense_1049/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
Λ	
ω
H__inference_dense_1049_layer_call_and_return_conditional_losses_79805040

inputs0
matmul_readvariableop_resource:<2-
biasadd_readvariableop_resource:2
identity’BiasAdd/ReadVariableOp’MatMul/ReadVariableOpt
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:<2*
dtype0i
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2r
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:2*
dtype0v
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2_
IdentityIdentityBiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2w
NoOpNoOp^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime**
_input_shapes
:?????????<: : 20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
?
	input_3502
serving_default_input_350:0?????????<>

dense_10490
StatefulPartitionedCall:0?????????2tensorflow/serving/predict:N
Ϋ
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
	optimizer
	variables
trainable_variables
regularization_losses
	keras_api
	__call__
*
&call_and_return_all_conditional_losses
_default_save_signature

signatures"
_tf_keras_sequential
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
trainable_variables
regularization_losses
	keras_api
__call__
*&call_and_return_all_conditional_losses"
_tf_keras_layer
»

kernel
bias
	variables
 trainable_variables
!regularization_losses
"	keras_api
#__call__
*$&call_and_return_all_conditional_losses"
_tf_keras_layer
Ώ
%iter

&beta_1

'beta_2
	(decay
)learning_ratemDmEmFmGmHmIvJvKvLvMvNvO"
	optimizer
J
0
1
2
3
4
5"
trackable_list_wrapper
J
0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
Κ
*non_trainable_variables

+layers
,metrics
-layer_regularization_losses
.layer_metrics
	variables
trainable_variables
regularization_losses
	__call__
_default_save_signature
*
&call_and_return_all_conditional_losses
&
"call_and_return_conditional_losses"
_generic_user_object
2
1__inference_sequential_349_layer_call_fn_79805062
1__inference_sequential_349_layer_call_fn_79805223
1__inference_sequential_349_layer_call_fn_79805240
1__inference_sequential_349_layer_call_fn_79805162ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ώ2ϋ
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805264
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805288
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805181
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805200ΐ
·²³
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsͺ 
annotationsͺ *
 
ΠBΝ
#__inference__wrapped_model_79804989	input_350"
²
FullArgSpec
args 
varargsjargs
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
,
/serving_default"
signature_map
#:!<<2dense_1047/kernel
:<2dense_1047/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
0non_trainable_variables

1layers
2metrics
3layer_regularization_losses
4layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Χ2Τ
-__inference_dense_1047_layer_call_fn_79805316’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ς2ο
H__inference_dense_1047_layer_call_and_return_conditional_losses_79805327’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
#:!<<2dense_1048/kernel
:<2dense_1048/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
5non_trainable_variables

6layers
7metrics
8layer_regularization_losses
9layer_metrics
	variables
trainable_variables
regularization_losses
__call__
*&call_and_return_all_conditional_losses
&"call_and_return_conditional_losses"
_generic_user_object
Χ2Τ
-__inference_dense_1048_layer_call_fn_79805336’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ς2ο
H__inference_dense_1048_layer_call_and_return_conditional_losses_79805347’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
#:!<22dense_1049/kernel
:22dense_1049/bias
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
­
:non_trainable_variables

;layers
<metrics
=layer_regularization_losses
>layer_metrics
	variables
 trainable_variables
!regularization_losses
#__call__
*$&call_and_return_all_conditional_losses
&$"call_and_return_conditional_losses"
_generic_user_object
Χ2Τ
-__inference_dense_1049_layer_call_fn_79805356’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
ς2ο
H__inference_dense_1049_layer_call_and_return_conditional_losses_79805366’
²
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
'
?0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
ΟBΜ
&__inference_signature_wrapper_79805307	input_350"
²
FullArgSpec
args 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsͺ *
 
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
N
	@total
	Acount
B	variables
C	keras_api"
_tf_keras_metric
:  (2total
:  (2count
.
@0
A1"
trackable_list_wrapper
-
B	variables"
_generic_user_object
(:&<<2Adam/dense_1047/kernel/m
": <2Adam/dense_1047/bias/m
(:&<<2Adam/dense_1048/kernel/m
": <2Adam/dense_1048/bias/m
(:&<22Adam/dense_1049/kernel/m
": 22Adam/dense_1049/bias/m
(:&<<2Adam/dense_1047/kernel/v
": <2Adam/dense_1047/bias/v
(:&<<2Adam/dense_1048/kernel/v
": <2Adam/dense_1048/bias/v
(:&<22Adam/dense_1049/kernel/v
": 22Adam/dense_1049/bias/v
#__inference__wrapped_model_79804989u2’/
(’%
# 
	input_350?????????<
ͺ "7ͺ4
2

dense_1049$!

dense_1049?????????2¨
H__inference_dense_1047_layer_call_and_return_conditional_losses_79805327\/’,
%’"
 
inputs?????????<
ͺ "%’"

0?????????<
 
-__inference_dense_1047_layer_call_fn_79805316O/’,
%’"
 
inputs?????????<
ͺ "?????????<¨
H__inference_dense_1048_layer_call_and_return_conditional_losses_79805347\/’,
%’"
 
inputs?????????<
ͺ "%’"

0?????????<
 
-__inference_dense_1048_layer_call_fn_79805336O/’,
%’"
 
inputs?????????<
ͺ "?????????<¨
H__inference_dense_1049_layer_call_and_return_conditional_losses_79805366\/’,
%’"
 
inputs?????????<
ͺ "%’"

0?????????2
 
-__inference_dense_1049_layer_call_fn_79805356O/’,
%’"
 
inputs?????????<
ͺ "?????????2»
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805181k:’7
0’-
# 
	input_350?????????<
p 

 
ͺ "%’"

0?????????2
 »
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805200k:’7
0’-
# 
	input_350?????????<
p

 
ͺ "%’"

0?????????2
 Έ
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805264h7’4
-’*
 
inputs?????????<
p 

 
ͺ "%’"

0?????????2
 Έ
L__inference_sequential_349_layer_call_and_return_conditional_losses_79805288h7’4
-’*
 
inputs?????????<
p

 
ͺ "%’"

0?????????2
 
1__inference_sequential_349_layer_call_fn_79805062^:’7
0’-
# 
	input_350?????????<
p 

 
ͺ "?????????2
1__inference_sequential_349_layer_call_fn_79805162^:’7
0’-
# 
	input_350?????????<
p

 
ͺ "?????????2
1__inference_sequential_349_layer_call_fn_79805223[7’4
-’*
 
inputs?????????<
p 

 
ͺ "?????????2
1__inference_sequential_349_layer_call_fn_79805240[7’4
-’*
 
inputs?????????<
p

 
ͺ "?????????2­
&__inference_signature_wrapper_79805307?’<
’ 
5ͺ2
0
	input_350# 
	input_350?????????<"7ͺ4
2

dense_1049$!

dense_1049?????????2