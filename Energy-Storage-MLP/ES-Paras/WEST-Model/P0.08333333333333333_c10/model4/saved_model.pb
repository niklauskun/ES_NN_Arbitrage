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
dense_1002/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*"
shared_namedense_1002/kernel
w
%dense_1002/kernel/Read/ReadVariableOpReadVariableOpdense_1002/kernel*
_output_shapes

:<<*
dtype0
v
dense_1002/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<* 
shared_namedense_1002/bias
o
#dense_1002/bias/Read/ReadVariableOpReadVariableOpdense_1002/bias*
_output_shapes
:<*
dtype0
~
dense_1003/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*"
shared_namedense_1003/kernel
w
%dense_1003/kernel/Read/ReadVariableOpReadVariableOpdense_1003/kernel*
_output_shapes

:<<*
dtype0
v
dense_1003/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:<* 
shared_namedense_1003/bias
o
#dense_1003/bias/Read/ReadVariableOpReadVariableOpdense_1003/bias*
_output_shapes
:<*
dtype0
~
dense_1004/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*"
shared_namedense_1004/kernel
w
%dense_1004/kernel/Read/ReadVariableOpReadVariableOpdense_1004/kernel*
_output_shapes

:<2*
dtype0
v
dense_1004/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:2* 
shared_namedense_1004/bias
o
#dense_1004/bias/Read/ReadVariableOpReadVariableOpdense_1004/bias*
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
Adam/dense_1002/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*)
shared_nameAdam/dense_1002/kernel/m

,Adam/dense_1002/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1002/kernel/m*
_output_shapes

:<<*
dtype0

Adam/dense_1002/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameAdam/dense_1002/bias/m
}
*Adam/dense_1002/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1002/bias/m*
_output_shapes
:<*
dtype0

Adam/dense_1003/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*)
shared_nameAdam/dense_1003/kernel/m

,Adam/dense_1003/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1003/kernel/m*
_output_shapes

:<<*
dtype0

Adam/dense_1003/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameAdam/dense_1003/bias/m
}
*Adam/dense_1003/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1003/bias/m*
_output_shapes
:<*
dtype0

Adam/dense_1004/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*)
shared_nameAdam/dense_1004/kernel/m

,Adam/dense_1004/kernel/m/Read/ReadVariableOpReadVariableOpAdam/dense_1004/kernel/m*
_output_shapes

:<2*
dtype0

Adam/dense_1004/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameAdam/dense_1004/bias/m
}
*Adam/dense_1004/bias/m/Read/ReadVariableOpReadVariableOpAdam/dense_1004/bias/m*
_output_shapes
:2*
dtype0

Adam/dense_1002/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*)
shared_nameAdam/dense_1002/kernel/v

,Adam/dense_1002/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1002/kernel/v*
_output_shapes

:<<*
dtype0

Adam/dense_1002/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameAdam/dense_1002/bias/v
}
*Adam/dense_1002/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1002/bias/v*
_output_shapes
:<*
dtype0

Adam/dense_1003/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<<*)
shared_nameAdam/dense_1003/kernel/v

,Adam/dense_1003/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1003/kernel/v*
_output_shapes

:<<*
dtype0

Adam/dense_1003/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:<*'
shared_nameAdam/dense_1003/bias/v
}
*Adam/dense_1003/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1003/bias/v*
_output_shapes
:<*
dtype0

Adam/dense_1004/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape
:<2*)
shared_nameAdam/dense_1004/kernel/v

,Adam/dense_1004/kernel/v/Read/ReadVariableOpReadVariableOpAdam/dense_1004/kernel/v*
_output_shapes

:<2*
dtype0

Adam/dense_1004/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:2*'
shared_nameAdam/dense_1004/bias/v
}
*Adam/dense_1004/bias/v/Read/ReadVariableOpReadVariableOpAdam/dense_1004/bias/v*
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
VARIABLE_VALUEdense_1002/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_1002/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_1003/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_1003/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEdense_1004/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE*
]W
VARIABLE_VALUEdense_1004/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE*
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
VARIABLE_VALUEAdam/dense_1002/kernel/mRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1002/bias/mPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1003/kernel/mRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1003/bias/mPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1004/kernel/mRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1004/bias/mPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1002/kernel/vRlayer_with_weights-0/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1002/bias/vPlayer_with_weights-0/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1003/kernel/vRlayer_with_weights-1/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1003/bias/vPlayer_with_weights-1/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
~
VARIABLE_VALUEAdam/dense_1004/kernel/vRlayer_with_weights-2/kernel/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
z
VARIABLE_VALUEAdam/dense_1004/bias/vPlayer_with_weights-2/bias/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE*
|
serving_default_input_335Placeholder*'
_output_shapes
:?????????<*
dtype0*
shape:?????????<
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_335dense_1002/kerneldense_1002/biasdense_1003/kerneldense_1003/biasdense_1004/kerneldense_1004/bias*
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
&__inference_signature_wrapper_76383792
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 


StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename%dense_1002/kernel/Read/ReadVariableOp#dense_1002/bias/Read/ReadVariableOp%dense_1003/kernel/Read/ReadVariableOp#dense_1003/bias/Read/ReadVariableOp%dense_1004/kernel/Read/ReadVariableOp#dense_1004/bias/Read/ReadVariableOpAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOp,Adam/dense_1002/kernel/m/Read/ReadVariableOp*Adam/dense_1002/bias/m/Read/ReadVariableOp,Adam/dense_1003/kernel/m/Read/ReadVariableOp*Adam/dense_1003/bias/m/Read/ReadVariableOp,Adam/dense_1004/kernel/m/Read/ReadVariableOp*Adam/dense_1004/bias/m/Read/ReadVariableOp,Adam/dense_1002/kernel/v/Read/ReadVariableOp*Adam/dense_1002/bias/v/Read/ReadVariableOp,Adam/dense_1003/kernel/v/Read/ReadVariableOp*Adam/dense_1003/bias/v/Read/ReadVariableOp,Adam/dense_1004/kernel/v/Read/ReadVariableOp*Adam/dense_1004/bias/v/Read/ReadVariableOpConst*&
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
!__inference__traced_save_76383949

StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense_1002/kerneldense_1002/biasdense_1003/kerneldense_1003/biasdense_1004/kerneldense_1004/bias	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_ratetotalcountAdam/dense_1002/kernel/mAdam/dense_1002/bias/mAdam/dense_1003/kernel/mAdam/dense_1003/bias/mAdam/dense_1004/kernel/mAdam/dense_1004/bias/mAdam/dense_1002/kernel/vAdam/dense_1002/bias/vAdam/dense_1003/kernel/vAdam/dense_1003/bias/vAdam/dense_1004/kernel/vAdam/dense_1004/bias/v*%
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
$__inference__traced_restore_76384034Θ
Κ

-__inference_dense_1003_layer_call_fn_76383821

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
H__inference_dense_1003_layer_call_and_return_conditional_losses_76383509o
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
Λ	
ω
H__inference_dense_1004_layer_call_and_return_conditional_losses_76383851

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
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383615

inputs%
dense_1002_76383599:<<!
dense_1002_76383601:<%
dense_1003_76383604:<<!
dense_1003_76383606:<%
dense_1004_76383609:<2!
dense_1004_76383611:2
identity’"dense_1002/StatefulPartitionedCall’"dense_1003/StatefulPartitionedCall’"dense_1004/StatefulPartitionedCallώ
"dense_1002/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1002_76383599dense_1002_76383601*
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
H__inference_dense_1002_layer_call_and_return_conditional_losses_76383492£
"dense_1003/StatefulPartitionedCallStatefulPartitionedCall+dense_1002/StatefulPartitionedCall:output:0dense_1003_76383604dense_1003_76383606*
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
H__inference_dense_1003_layer_call_and_return_conditional_losses_76383509£
"dense_1004/StatefulPartitionedCallStatefulPartitionedCall+dense_1003/StatefulPartitionedCall:output:0dense_1004_76383609dense_1004_76383611*
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
H__inference_dense_1004_layer_call_and_return_conditional_losses_76383525z
IdentityIdentity+dense_1004/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2΅
NoOpNoOp#^dense_1002/StatefulPartitionedCall#^dense_1003/StatefulPartitionedCall#^dense_1004/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 2H
"dense_1002/StatefulPartitionedCall"dense_1002/StatefulPartitionedCall2H
"dense_1003/StatefulPartitionedCall"dense_1003/StatefulPartitionedCall2H
"dense_1004/StatefulPartitionedCall"dense_1004/StatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
Κ

-__inference_dense_1004_layer_call_fn_76383841

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
H__inference_dense_1004_layer_call_and_return_conditional_losses_76383525o
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
	

1__inference_sequential_334_layer_call_fn_76383647
	input_335
unknown:<<
	unknown_0:<
	unknown_1:<<
	unknown_2:<
	unknown_3:<2
	unknown_4:2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	input_335unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383615o
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
_user_specified_name	input_335
Λ	
ω
H__inference_dense_1004_layer_call_and_return_conditional_losses_76383525

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
ϊ

1__inference_sequential_334_layer_call_fn_76383708

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
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383532o
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
H__inference_dense_1002_layer_call_and_return_conditional_losses_76383492

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
	

1__inference_sequential_334_layer_call_fn_76383547
	input_335
unknown:<<
	unknown_0:<
	unknown_1:<<
	unknown_2:<
	unknown_3:<2
	unknown_4:2
identity’StatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCall	input_335unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383532o
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
_user_specified_name	input_335


ω
H__inference_dense_1003_layer_call_and_return_conditional_losses_76383509

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
θ
Έ
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383532

inputs%
dense_1002_76383493:<<!
dense_1002_76383495:<%
dense_1003_76383510:<<!
dense_1003_76383512:<%
dense_1004_76383526:<2!
dense_1004_76383528:2
identity’"dense_1002/StatefulPartitionedCall’"dense_1003/StatefulPartitionedCall’"dense_1004/StatefulPartitionedCallώ
"dense_1002/StatefulPartitionedCallStatefulPartitionedCallinputsdense_1002_76383493dense_1002_76383495*
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
H__inference_dense_1002_layer_call_and_return_conditional_losses_76383492£
"dense_1003/StatefulPartitionedCallStatefulPartitionedCall+dense_1002/StatefulPartitionedCall:output:0dense_1003_76383510dense_1003_76383512*
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
H__inference_dense_1003_layer_call_and_return_conditional_losses_76383509£
"dense_1004/StatefulPartitionedCallStatefulPartitionedCall+dense_1003/StatefulPartitionedCall:output:0dense_1004_76383526dense_1004_76383528*
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
H__inference_dense_1004_layer_call_and_return_conditional_losses_76383525z
IdentityIdentity+dense_1004/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2΅
NoOpNoOp#^dense_1002/StatefulPartitionedCall#^dense_1003/StatefulPartitionedCall#^dense_1004/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 2H
"dense_1002/StatefulPartitionedCall"dense_1002/StatefulPartitionedCall2H
"dense_1003/StatefulPartitionedCall"dense_1003/StatefulPartitionedCall2H
"dense_1004/StatefulPartitionedCall"dense_1004/StatefulPartitionedCall:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
Ϊ9
ΰ

!__inference__traced_save_76383949
file_prefix0
,savev2_dense_1002_kernel_read_readvariableop.
*savev2_dense_1002_bias_read_readvariableop0
,savev2_dense_1003_kernel_read_readvariableop.
*savev2_dense_1003_bias_read_readvariableop0
,savev2_dense_1004_kernel_read_readvariableop.
*savev2_dense_1004_bias_read_readvariableop(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableop7
3savev2_adam_dense_1002_kernel_m_read_readvariableop5
1savev2_adam_dense_1002_bias_m_read_readvariableop7
3savev2_adam_dense_1003_kernel_m_read_readvariableop5
1savev2_adam_dense_1003_bias_m_read_readvariableop7
3savev2_adam_dense_1004_kernel_m_read_readvariableop5
1savev2_adam_dense_1004_bias_m_read_readvariableop7
3savev2_adam_dense_1002_kernel_v_read_readvariableop5
1savev2_adam_dense_1002_bias_v_read_readvariableop7
3savev2_adam_dense_1003_kernel_v_read_readvariableop5
1savev2_adam_dense_1003_bias_v_read_readvariableop7
3savev2_adam_dense_1004_kernel_v_read_readvariableop5
1savev2_adam_dense_1004_bias_v_read_readvariableop
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

SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0,savev2_dense_1002_kernel_read_readvariableop*savev2_dense_1002_bias_read_readvariableop,savev2_dense_1003_kernel_read_readvariableop*savev2_dense_1003_bias_read_readvariableop,savev2_dense_1004_kernel_read_readvariableop*savev2_dense_1004_bias_read_readvariableop$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableop3savev2_adam_dense_1002_kernel_m_read_readvariableop1savev2_adam_dense_1002_bias_m_read_readvariableop3savev2_adam_dense_1003_kernel_m_read_readvariableop1savev2_adam_dense_1003_bias_m_read_readvariableop3savev2_adam_dense_1004_kernel_m_read_readvariableop1savev2_adam_dense_1004_bias_m_read_readvariableop3savev2_adam_dense_1002_kernel_v_read_readvariableop1savev2_adam_dense_1002_bias_v_read_readvariableop3savev2_adam_dense_1003_kernel_v_read_readvariableop1savev2_adam_dense_1003_bias_v_read_readvariableop3savev2_adam_dense_1004_kernel_v_read_readvariableop1savev2_adam_dense_1004_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
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
Γ
₯
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383749

inputs;
)dense_1002_matmul_readvariableop_resource:<<8
*dense_1002_biasadd_readvariableop_resource:<;
)dense_1003_matmul_readvariableop_resource:<<8
*dense_1003_biasadd_readvariableop_resource:<;
)dense_1004_matmul_readvariableop_resource:<28
*dense_1004_biasadd_readvariableop_resource:2
identity’!dense_1002/BiasAdd/ReadVariableOp’ dense_1002/MatMul/ReadVariableOp’!dense_1003/BiasAdd/ReadVariableOp’ dense_1003/MatMul/ReadVariableOp’!dense_1004/BiasAdd/ReadVariableOp’ dense_1004/MatMul/ReadVariableOp
 dense_1002/MatMul/ReadVariableOpReadVariableOp)dense_1002_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype0
dense_1002/MatMulMatMulinputs(dense_1002/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<
!dense_1002/BiasAdd/ReadVariableOpReadVariableOp*dense_1002_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0
dense_1002/BiasAddBiasAdddense_1002/MatMul:product:0)dense_1002/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<f
dense_1002/ReluReludense_1002/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<
 dense_1003/MatMul/ReadVariableOpReadVariableOp)dense_1003_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype0
dense_1003/MatMulMatMuldense_1002/Relu:activations:0(dense_1003/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<
!dense_1003/BiasAdd/ReadVariableOpReadVariableOp*dense_1003_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0
dense_1003/BiasAddBiasAdddense_1003/MatMul:product:0)dense_1003/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<f
dense_1003/ReluReludense_1003/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<
 dense_1004/MatMul/ReadVariableOpReadVariableOp)dense_1004_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype0
dense_1004/MatMulMatMuldense_1003/Relu:activations:0(dense_1004/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
!dense_1004/BiasAdd/ReadVariableOpReadVariableOp*dense_1004_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_1004/BiasAddBiasAdddense_1004/MatMul:product:0)dense_1004/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2j
IdentityIdentitydense_1004/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2
NoOpNoOp"^dense_1002/BiasAdd/ReadVariableOp!^dense_1002/MatMul/ReadVariableOp"^dense_1003/BiasAdd/ReadVariableOp!^dense_1003/MatMul/ReadVariableOp"^dense_1004/BiasAdd/ReadVariableOp!^dense_1004/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 2F
!dense_1002/BiasAdd/ReadVariableOp!dense_1002/BiasAdd/ReadVariableOp2D
 dense_1002/MatMul/ReadVariableOp dense_1002/MatMul/ReadVariableOp2F
!dense_1003/BiasAdd/ReadVariableOp!dense_1003/BiasAdd/ReadVariableOp2D
 dense_1003/MatMul/ReadVariableOp dense_1003/MatMul/ReadVariableOp2F
!dense_1004/BiasAdd/ReadVariableOp!dense_1004/BiasAdd/ReadVariableOp2D
 dense_1004/MatMul/ReadVariableOp dense_1004/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
Κ

-__inference_dense_1002_layer_call_fn_76383801

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
H__inference_dense_1002_layer_call_and_return_conditional_losses_76383492o
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
f

$__inference__traced_restore_76384034
file_prefix4
"assignvariableop_dense_1002_kernel:<<0
"assignvariableop_1_dense_1002_bias:<6
$assignvariableop_2_dense_1003_kernel:<<0
"assignvariableop_3_dense_1003_bias:<6
$assignvariableop_4_dense_1004_kernel:<20
"assignvariableop_5_dense_1004_bias:2&
assignvariableop_6_adam_iter:	 (
assignvariableop_7_adam_beta_1: (
assignvariableop_8_adam_beta_2: '
assignvariableop_9_adam_decay: 0
&assignvariableop_10_adam_learning_rate: #
assignvariableop_11_total: #
assignvariableop_12_count: >
,assignvariableop_13_adam_dense_1002_kernel_m:<<8
*assignvariableop_14_adam_dense_1002_bias_m:<>
,assignvariableop_15_adam_dense_1003_kernel_m:<<8
*assignvariableop_16_adam_dense_1003_bias_m:<>
,assignvariableop_17_adam_dense_1004_kernel_m:<28
*assignvariableop_18_adam_dense_1004_bias_m:2>
,assignvariableop_19_adam_dense_1002_kernel_v:<<8
*assignvariableop_20_adam_dense_1002_bias_v:<>
,assignvariableop_21_adam_dense_1003_kernel_v:<<8
*assignvariableop_22_adam_dense_1003_bias_v:<>
,assignvariableop_23_adam_dense_1004_kernel_v:<28
*assignvariableop_24_adam_dense_1004_bias_v:2
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
AssignVariableOpAssignVariableOp"assignvariableop_dense_1002_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_1AssignVariableOp"assignvariableop_1_dense_1002_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_2AssignVariableOp$assignvariableop_2_dense_1003_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_3AssignVariableOp"assignvariableop_3_dense_1003_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_4AssignVariableOp$assignvariableop_4_dense_1004_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype0]

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_5AssignVariableOp"assignvariableop_5_dense_1004_biasIdentity_5:output:0"/device:CPU:0*
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
AssignVariableOp_13AssignVariableOp,assignvariableop_13_adam_dense_1002_kernel_mIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_14AssignVariableOp*assignvariableop_14_adam_dense_1002_bias_mIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_15AssignVariableOp,assignvariableop_15_adam_dense_1003_kernel_mIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_16AssignVariableOp*assignvariableop_16_adam_dense_1003_bias_mIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_17AssignVariableOp,assignvariableop_17_adam_dense_1004_kernel_mIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_18AssignVariableOp*assignvariableop_18_adam_dense_1004_bias_mIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_19AssignVariableOp,assignvariableop_19_adam_dense_1002_kernel_vIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_20AssignVariableOp*assignvariableop_20_adam_dense_1002_bias_vIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_21AssignVariableOp,assignvariableop_21_adam_dense_1003_kernel_vIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_22AssignVariableOp*assignvariableop_22_adam_dense_1003_bias_vIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_23AssignVariableOp,assignvariableop_23_adam_dense_1004_kernel_vIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype0_
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:
AssignVariableOp_24AssignVariableOp*assignvariableop_24_adam_dense_1004_bias_vIdentity_24:output:0"/device:CPU:0*
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
ρ
»
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383666
	input_335%
dense_1002_76383650:<<!
dense_1002_76383652:<%
dense_1003_76383655:<<!
dense_1003_76383657:<%
dense_1004_76383660:<2!
dense_1004_76383662:2
identity’"dense_1002/StatefulPartitionedCall’"dense_1003/StatefulPartitionedCall’"dense_1004/StatefulPartitionedCall
"dense_1002/StatefulPartitionedCallStatefulPartitionedCall	input_335dense_1002_76383650dense_1002_76383652*
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
H__inference_dense_1002_layer_call_and_return_conditional_losses_76383492£
"dense_1003/StatefulPartitionedCallStatefulPartitionedCall+dense_1002/StatefulPartitionedCall:output:0dense_1003_76383655dense_1003_76383657*
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
H__inference_dense_1003_layer_call_and_return_conditional_losses_76383509£
"dense_1004/StatefulPartitionedCallStatefulPartitionedCall+dense_1003/StatefulPartitionedCall:output:0dense_1004_76383660dense_1004_76383662*
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
H__inference_dense_1004_layer_call_and_return_conditional_losses_76383525z
IdentityIdentity+dense_1004/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2΅
NoOpNoOp#^dense_1002/StatefulPartitionedCall#^dense_1003/StatefulPartitionedCall#^dense_1004/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 2H
"dense_1002/StatefulPartitionedCall"dense_1002/StatefulPartitionedCall2H
"dense_1003/StatefulPartitionedCall"dense_1003/StatefulPartitionedCall2H
"dense_1004/StatefulPartitionedCall"dense_1004/StatefulPartitionedCall:R N
'
_output_shapes
:?????????<
#
_user_specified_name	input_335


ω
H__inference_dense_1003_layer_call_and_return_conditional_losses_76383832

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


ω
H__inference_dense_1002_layer_call_and_return_conditional_losses_76383812

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
ρ
»
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383685
	input_335%
dense_1002_76383669:<<!
dense_1002_76383671:<%
dense_1003_76383674:<<!
dense_1003_76383676:<%
dense_1004_76383679:<2!
dense_1004_76383681:2
identity’"dense_1002/StatefulPartitionedCall’"dense_1003/StatefulPartitionedCall’"dense_1004/StatefulPartitionedCall
"dense_1002/StatefulPartitionedCallStatefulPartitionedCall	input_335dense_1002_76383669dense_1002_76383671*
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
H__inference_dense_1002_layer_call_and_return_conditional_losses_76383492£
"dense_1003/StatefulPartitionedCallStatefulPartitionedCall+dense_1002/StatefulPartitionedCall:output:0dense_1003_76383674dense_1003_76383676*
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
H__inference_dense_1003_layer_call_and_return_conditional_losses_76383509£
"dense_1004/StatefulPartitionedCallStatefulPartitionedCall+dense_1003/StatefulPartitionedCall:output:0dense_1004_76383679dense_1004_76383681*
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
H__inference_dense_1004_layer_call_and_return_conditional_losses_76383525z
IdentityIdentity+dense_1004/StatefulPartitionedCall:output:0^NoOp*
T0*'
_output_shapes
:?????????2΅
NoOpNoOp#^dense_1002/StatefulPartitionedCall#^dense_1003/StatefulPartitionedCall#^dense_1004/StatefulPartitionedCall*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 2H
"dense_1002/StatefulPartitionedCall"dense_1002/StatefulPartitionedCall2H
"dense_1003/StatefulPartitionedCall"dense_1003/StatefulPartitionedCall2H
"dense_1004/StatefulPartitionedCall"dense_1004/StatefulPartitionedCall:R N
'
_output_shapes
:?????????<
#
_user_specified_name	input_335
ϊ

1__inference_sequential_334_layer_call_fn_76383725

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
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383615o
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
ζ#
³
#__inference__wrapped_model_76383474
	input_335J
8sequential_334_dense_1002_matmul_readvariableop_resource:<<G
9sequential_334_dense_1002_biasadd_readvariableop_resource:<J
8sequential_334_dense_1003_matmul_readvariableop_resource:<<G
9sequential_334_dense_1003_biasadd_readvariableop_resource:<J
8sequential_334_dense_1004_matmul_readvariableop_resource:<2G
9sequential_334_dense_1004_biasadd_readvariableop_resource:2
identity’0sequential_334/dense_1002/BiasAdd/ReadVariableOp’/sequential_334/dense_1002/MatMul/ReadVariableOp’0sequential_334/dense_1003/BiasAdd/ReadVariableOp’/sequential_334/dense_1003/MatMul/ReadVariableOp’0sequential_334/dense_1004/BiasAdd/ReadVariableOp’/sequential_334/dense_1004/MatMul/ReadVariableOp¨
/sequential_334/dense_1002/MatMul/ReadVariableOpReadVariableOp8sequential_334_dense_1002_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype0 
 sequential_334/dense_1002/MatMulMatMul	input_3357sequential_334/dense_1002/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<¦
0sequential_334/dense_1002/BiasAdd/ReadVariableOpReadVariableOp9sequential_334_dense_1002_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0Δ
!sequential_334/dense_1002/BiasAddBiasAdd*sequential_334/dense_1002/MatMul:product:08sequential_334/dense_1002/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<
sequential_334/dense_1002/ReluRelu*sequential_334/dense_1002/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<¨
/sequential_334/dense_1003/MatMul/ReadVariableOpReadVariableOp8sequential_334_dense_1003_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype0Γ
 sequential_334/dense_1003/MatMulMatMul,sequential_334/dense_1002/Relu:activations:07sequential_334/dense_1003/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<¦
0sequential_334/dense_1003/BiasAdd/ReadVariableOpReadVariableOp9sequential_334_dense_1003_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0Δ
!sequential_334/dense_1003/BiasAddBiasAdd*sequential_334/dense_1003/MatMul:product:08sequential_334/dense_1003/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<
sequential_334/dense_1003/ReluRelu*sequential_334/dense_1003/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<¨
/sequential_334/dense_1004/MatMul/ReadVariableOpReadVariableOp8sequential_334_dense_1004_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype0Γ
 sequential_334/dense_1004/MatMulMatMul,sequential_334/dense_1003/Relu:activations:07sequential_334/dense_1004/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2¦
0sequential_334/dense_1004/BiasAdd/ReadVariableOpReadVariableOp9sequential_334_dense_1004_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0Δ
!sequential_334/dense_1004/BiasAddBiasAdd*sequential_334/dense_1004/MatMul:product:08sequential_334/dense_1004/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2y
IdentityIdentity*sequential_334/dense_1004/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2υ
NoOpNoOp1^sequential_334/dense_1002/BiasAdd/ReadVariableOp0^sequential_334/dense_1002/MatMul/ReadVariableOp1^sequential_334/dense_1003/BiasAdd/ReadVariableOp0^sequential_334/dense_1003/MatMul/ReadVariableOp1^sequential_334/dense_1004/BiasAdd/ReadVariableOp0^sequential_334/dense_1004/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 2d
0sequential_334/dense_1002/BiasAdd/ReadVariableOp0sequential_334/dense_1002/BiasAdd/ReadVariableOp2b
/sequential_334/dense_1002/MatMul/ReadVariableOp/sequential_334/dense_1002/MatMul/ReadVariableOp2d
0sequential_334/dense_1003/BiasAdd/ReadVariableOp0sequential_334/dense_1003/BiasAdd/ReadVariableOp2b
/sequential_334/dense_1003/MatMul/ReadVariableOp/sequential_334/dense_1003/MatMul/ReadVariableOp2d
0sequential_334/dense_1004/BiasAdd/ReadVariableOp0sequential_334/dense_1004/BiasAdd/ReadVariableOp2b
/sequential_334/dense_1004/MatMul/ReadVariableOp/sequential_334/dense_1004/MatMul/ReadVariableOp:R N
'
_output_shapes
:?????????<
#
_user_specified_name	input_335
Γ
₯
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383773

inputs;
)dense_1002_matmul_readvariableop_resource:<<8
*dense_1002_biasadd_readvariableop_resource:<;
)dense_1003_matmul_readvariableop_resource:<<8
*dense_1003_biasadd_readvariableop_resource:<;
)dense_1004_matmul_readvariableop_resource:<28
*dense_1004_biasadd_readvariableop_resource:2
identity’!dense_1002/BiasAdd/ReadVariableOp’ dense_1002/MatMul/ReadVariableOp’!dense_1003/BiasAdd/ReadVariableOp’ dense_1003/MatMul/ReadVariableOp’!dense_1004/BiasAdd/ReadVariableOp’ dense_1004/MatMul/ReadVariableOp
 dense_1002/MatMul/ReadVariableOpReadVariableOp)dense_1002_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype0
dense_1002/MatMulMatMulinputs(dense_1002/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<
!dense_1002/BiasAdd/ReadVariableOpReadVariableOp*dense_1002_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0
dense_1002/BiasAddBiasAdddense_1002/MatMul:product:0)dense_1002/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<f
dense_1002/ReluReludense_1002/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<
 dense_1003/MatMul/ReadVariableOpReadVariableOp)dense_1003_matmul_readvariableop_resource*
_output_shapes

:<<*
dtype0
dense_1003/MatMulMatMuldense_1002/Relu:activations:0(dense_1003/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<
!dense_1003/BiasAdd/ReadVariableOpReadVariableOp*dense_1003_biasadd_readvariableop_resource*
_output_shapes
:<*
dtype0
dense_1003/BiasAddBiasAdddense_1003/MatMul:product:0)dense_1003/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????<f
dense_1003/ReluReludense_1003/BiasAdd:output:0*
T0*'
_output_shapes
:?????????<
 dense_1004/MatMul/ReadVariableOpReadVariableOp)dense_1004_matmul_readvariableop_resource*
_output_shapes

:<2*
dtype0
dense_1004/MatMulMatMuldense_1003/Relu:activations:0(dense_1004/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
!dense_1004/BiasAdd/ReadVariableOpReadVariableOp*dense_1004_biasadd_readvariableop_resource*
_output_shapes
:2*
dtype0
dense_1004/BiasAddBiasAdddense_1004/MatMul:product:0)dense_1004/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2j
IdentityIdentitydense_1004/BiasAdd:output:0^NoOp*
T0*'
_output_shapes
:?????????2
NoOpNoOp"^dense_1002/BiasAdd/ReadVariableOp!^dense_1002/MatMul/ReadVariableOp"^dense_1003/BiasAdd/ReadVariableOp!^dense_1003/MatMul/ReadVariableOp"^dense_1004/BiasAdd/ReadVariableOp!^dense_1004/MatMul/ReadVariableOp*"
_acd_function_control_output(*
_output_shapes
 "
identityIdentity:output:0*(
_construction_contextkEagerRuntime*2
_input_shapes!
:?????????<: : : : : : 2F
!dense_1002/BiasAdd/ReadVariableOp!dense_1002/BiasAdd/ReadVariableOp2D
 dense_1002/MatMul/ReadVariableOp dense_1002/MatMul/ReadVariableOp2F
!dense_1003/BiasAdd/ReadVariableOp!dense_1003/BiasAdd/ReadVariableOp2D
 dense_1003/MatMul/ReadVariableOp dense_1003/MatMul/ReadVariableOp2F
!dense_1004/BiasAdd/ReadVariableOp!dense_1004/BiasAdd/ReadVariableOp2D
 dense_1004/MatMul/ReadVariableOp dense_1004/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????<
 
_user_specified_nameinputs
Ο

&__inference_signature_wrapper_76383792
	input_335
unknown:<<
	unknown_0:<
	unknown_1:<<
	unknown_2:<
	unknown_3:<2
	unknown_4:2
identity’StatefulPartitionedCallο
StatefulPartitionedCallStatefulPartitionedCall	input_335unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4*
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
#__inference__wrapped_model_76383474o
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
_user_specified_name	input_335"ΫL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*±
serving_default
?
	input_3352
serving_default_input_335:0?????????<>

dense_10040
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
1__inference_sequential_334_layer_call_fn_76383547
1__inference_sequential_334_layer_call_fn_76383708
1__inference_sequential_334_layer_call_fn_76383725
1__inference_sequential_334_layer_call_fn_76383647ΐ
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
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383749
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383773
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383666
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383685ΐ
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
#__inference__wrapped_model_76383474	input_335"
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
#:!<<2dense_1002/kernel
:<2dense_1002/bias
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
-__inference_dense_1002_layer_call_fn_76383801’
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
H__inference_dense_1002_layer_call_and_return_conditional_losses_76383812’
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
#:!<<2dense_1003/kernel
:<2dense_1003/bias
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
-__inference_dense_1003_layer_call_fn_76383821’
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
H__inference_dense_1003_layer_call_and_return_conditional_losses_76383832’
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
#:!<22dense_1004/kernel
:22dense_1004/bias
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
-__inference_dense_1004_layer_call_fn_76383841’
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
H__inference_dense_1004_layer_call_and_return_conditional_losses_76383851’
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
&__inference_signature_wrapper_76383792	input_335"
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
(:&<<2Adam/dense_1002/kernel/m
": <2Adam/dense_1002/bias/m
(:&<<2Adam/dense_1003/kernel/m
": <2Adam/dense_1003/bias/m
(:&<22Adam/dense_1004/kernel/m
": 22Adam/dense_1004/bias/m
(:&<<2Adam/dense_1002/kernel/v
": <2Adam/dense_1002/bias/v
(:&<<2Adam/dense_1003/kernel/v
": <2Adam/dense_1003/bias/v
(:&<22Adam/dense_1004/kernel/v
": 22Adam/dense_1004/bias/v
#__inference__wrapped_model_76383474u2’/
(’%
# 
	input_335?????????<
ͺ "7ͺ4
2

dense_1004$!

dense_1004?????????2¨
H__inference_dense_1002_layer_call_and_return_conditional_losses_76383812\/’,
%’"
 
inputs?????????<
ͺ "%’"

0?????????<
 
-__inference_dense_1002_layer_call_fn_76383801O/’,
%’"
 
inputs?????????<
ͺ "?????????<¨
H__inference_dense_1003_layer_call_and_return_conditional_losses_76383832\/’,
%’"
 
inputs?????????<
ͺ "%’"

0?????????<
 
-__inference_dense_1003_layer_call_fn_76383821O/’,
%’"
 
inputs?????????<
ͺ "?????????<¨
H__inference_dense_1004_layer_call_and_return_conditional_losses_76383851\/’,
%’"
 
inputs?????????<
ͺ "%’"

0?????????2
 
-__inference_dense_1004_layer_call_fn_76383841O/’,
%’"
 
inputs?????????<
ͺ "?????????2»
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383666k:’7
0’-
# 
	input_335?????????<
p 

 
ͺ "%’"

0?????????2
 »
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383685k:’7
0’-
# 
	input_335?????????<
p

 
ͺ "%’"

0?????????2
 Έ
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383749h7’4
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
L__inference_sequential_334_layer_call_and_return_conditional_losses_76383773h7’4
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
1__inference_sequential_334_layer_call_fn_76383547^:’7
0’-
# 
	input_335?????????<
p 

 
ͺ "?????????2
1__inference_sequential_334_layer_call_fn_76383647^:’7
0’-
# 
	input_335?????????<
p

 
ͺ "?????????2
1__inference_sequential_334_layer_call_fn_76383708[7’4
-’*
 
inputs?????????<
p 

 
ͺ "?????????2
1__inference_sequential_334_layer_call_fn_76383725[7’4
-’*
 
inputs?????????<
p

 
ͺ "?????????2­
&__inference_signature_wrapper_76383792?’<
’ 
5ͺ2
0
	input_335# 
	input_335?????????<"7ͺ4
2

dense_1004$!

dense_1004?????????2