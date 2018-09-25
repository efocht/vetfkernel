	.text
	.file	"libvetfkernel_BiasAdd.c"
	.globl	BiasAdd_NHWC
	.p2align	4
	.type	BiasAdd_NHWC,@function
BiasAdd_NHWC:
	st %s9, (,%s11)
	st %s10, 8(,%s11)
	st %s15, 24(,%s11)
	st %s16, 32(,%s11)
	or %s9, 0, %s11
	lea %s13, -176
	and %s13, %s13, (32)0
	lea.sl %s11, -1(%s11, %s13)
	brge.l %s11, %s8, .LBB0_14
	ld %s61, 24(,%s14)
	or %s62, 0, %s0
	lea %s63, 315
	shm.l %s63, (%s61)
	shm.l %s8, 8(%s61)
	shm.l %s11, 16(%s61)
	monc
	or %s0, 0, %s62
.LBB0_14:
	or %s34, 1, (0)1
	brlt.w %s3, %s34, .LBB0_12
	adds.w.zx %s35, %s6, (0)1
	adds.w.zx %s36, %s4, (0)1
	muls.w.sx %s37, %s6, %s5
	muls.w.sx %s37, %s37, %s4
	muls.w.sx %s38, %s6, %s4
	or %s39, 0, (0)1
	or %s40, 0, (0)1
	or %s41, 0, %s39
	or %s42, 0, %s39
.LBB0_2:
	brlt.w %s5, %s34, .LBB0_11
	brlt.w %s4, %s34, .LBB0_11
	or %s43, 0, %s41
	or %s44, 0, %s39
.LBB0_5:
	brlt.w %s6, %s34, .LBB0_10
	or %s45, 0, %s43
	or %s46, 0, %s40
.LBB0_7:
	adds.w.sx %s47, %s45, (0)1
	sll %s48, %s47, 2
	adds.l %s47, %s1, %s48
	adds.l %s48, %s0, %s48
	or %s49, 0, %s35
	or %s50, 0, %s2
.LBB0_8:
	ldu %s51, (,%s47)
	ldu %s52, (,%s50)
	fadd.s %s51, %s51, %s52
	stu %s51, (,%s48)
	lea %s50, 4(%s50)
	lea %s47, 4(%s47)
	lea %s49, -1(%s49)
	lea %s48, 4(%s48)
	brne.l %s49, %s40, .LBB0_8
	lea %s46, 1(%s46)
	adds.w.sx %s45, %s45, %s6
	brne.l %s46, %s36, .LBB0_7
.LBB0_10:
	lea %s44, 1(%s44)
	adds.w.sx %s43, %s43, %s38
	brne.w %s44, %s5, .LBB0_5
.LBB0_11:
	lea %s42, 1(%s42)
	adds.w.sx %s41, %s41, %s37
	brne.w %s42, %s3, .LBB0_2
.LBB0_12:
	or %s0, 0, (0)1
	or %s11, 0, %s9
	ld %s16, 32(,%s11)
	ld %s15, 24(,%s11)
	ld %s10, 8(,%s11)
	ld %s9, (,%s11)
	b.l (,%lr)
.Lfunc_end0:
	.size	BiasAdd_NHWC, .Lfunc_end0-BiasAdd_NHWC

	.globl	BiasAdd_NCHW
	.p2align	4
	.type	BiasAdd_NCHW,@function
BiasAdd_NCHW:
	st %s9, (,%s11)
	st %s10, 8(,%s11)
	st %s15, 24(,%s11)
	st %s16, 32(,%s11)
	or %s9, 0, %s11
	lea %s13, -176
	and %s13, %s13, (32)0
	lea.sl %s11, -1(%s11, %s13)
	brge.l %s11, %s8, .LBB1_11
	ld %s61, 24(,%s14)
	or %s62, 0, %s0
	lea %s63, 315
	shm.l %s63, (%s61)
	shm.l %s8, 8(%s61)
	shm.l %s11, 16(%s61)
	monc
	or %s0, 0, %s62
.LBB1_11:
	or %s34, 1, (0)1
	brlt.w %s3, %s34, .LBB1_9
	brlt.w %s6, %s34, .LBB1_9
	muls.w.sx %s35, %s5, %s4
	adds.w.zx %s36, %s35, (0)1
	adds.w.zx %s37, %s6, (0)1
	muls.w.sx %s38, %s6, %s5
	muls.w.sx %s38, %s38, %s4
	or %s39, 0, (0)1
	or %s40, 0, (0)1
	or %s41, 0, %s39
.LBB1_3:
	brlt.w %s35, %s34, .LBB1_8
	or %s42, 0, %s39
	or %s43, 0, %s40
.LBB1_5:
	sll %s44, %s43, 2
	adds.l %s44, %s2, %s44
	or %s45, 0, %s36
	or %s46, 0, %s42
.LBB1_6:
	adds.w.sx %s47, %s46, (0)1
	sll %s47, %s47, 2
	adds.l %s48, %s1, %s47
	ldu %s48, (,%s48)
	ldu %s49, (,%s44)
	fadd.s %s48, %s48, %s49
	adds.l %s47, %s0, %s47
	stu %s48, (,%s47)
	lea %s45, -1(%s45)
	lea %s46, 1(%s46)
	brne.l %s45, %s40, .LBB1_6
	lea %s43, 1(%s43)
	adds.w.sx %s42, %s42, %s35
	brne.l %s43, %s37, .LBB1_5
.LBB1_8:
	lea %s41, 1(%s41)
	adds.w.sx %s39, %s39, %s38
	brne.w %s41, %s3, .LBB1_3
.LBB1_9:
	or %s0, 0, (0)1
	or %s11, 0, %s9
	ld %s16, 32(,%s11)
	ld %s15, 24(,%s11)
	ld %s10, 8(,%s11)
	ld %s9, (,%s11)
	b.l (,%lr)
.Lfunc_end1:
	.size	BiasAdd_NCHW, .Lfunc_end1-BiasAdd_NCHW


	.ident	"clang version 8.0.0 (git@socsv218.svp.cl.nec.co.jp:ve-llvm/clang.git fbb6d58b08faee86964f281717acbb628686b873) (llvm/llvm.git 480dd04aa3e76007f634ee0dbb900cb44fe53c11)"
	.section	".note.GNU-stack","",@progbits
