//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// Also available under a BSD-style license. See LICENSE.
//
//===----------------------------------------------------------------------===//

#include "torch-mlir/Conversion/TorchToStablehlo/TorchToStablehlo.h"

#include "../PassDetail.h"
#include "PopulatePatterns.h"
#include "mlir/IR/ValueRange.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "stablehlo/dialect/ChloOps.h"
#include "stablehlo/dialect/StablehloOps.h"
#include "torch-mlir/Conversion/TorchToStablehlo/StablehloLegalizeUtils.h"
#include "torch-mlir/Conversion/Utils/Utils.h"
#include "torch-mlir/Dialect/Torch/IR/TorchDialect.h"
#include "torch-mlir/Dialect/Torch/IR/TorchOps.h"
#include "torch-mlir/Dialect/Torch/Utils/Utils.h"
#include "torch-mlir/Dialect/TorchConversion/IR/TorchConversionOps.h"
#include <iostream>

using namespace mlir;
using namespace mlir::torch;
using namespace mlir::torch::Torch;
using namespace mlir::torch::torch_to_stablehlo;

// AtenEmbeddingPaddingIdxOp
// SUM mode == integer 0
// Sums bags of embeddings together from a weight tensor based on an index and
// offset Vector. Example arguments weight = [[1, 3, 5, 3],
//           [3, 4, 2, 1],
//           [2, 2, 3, 2],
//           [0, 4, 2, 1]]
//
// indices = [0, 2, 3, 1, 2, 3, 2, 1, 0, 1]
// offsets = [0, 3, 5]
//
// output_tensor = initZeroTensor(offsets_length, embedding_size)
//
// for i in range(offsets_length):         <- dim0
//     for j in range(indices_length):     <- dim1
//         for k in range(embedding_size): <- dim2
//             if(offsets[i] <= j and j < offsets[i+1]):
//                 output_tensor[i][k] = output_tensor[i][k] +
//                 weight[indices[j]][k]
//             else:
//                 break
//
// Indexing maps for linalg::Generic ops
//
//
// indices_indexing_map  = (d0, d1, d2) -> (d1)
// offset_indexing_map   = (d0, d1, d2) -> (d0)
// output_indexing_map   = (d0, d1, d2) -> (d0, d2)
//
// TODO: Find an optimal lowering.
//       current lowering is not optimal for bags of large embeddings.
//       Since it traverses the output tensor multiple times. 
//      
//
static Value createInitialValueForAtenIndirectDataMovementOp(Operation *op, Type elementTy,
                                                PatternRewriter &rewriter) {
  auto constType = RankedTensorType::get({}, elementTy);
  // EmbeddingBagPaddingIdx
  if (isa<AtenEmbeddingBagPaddingIdxOp>(op)) {
    if (elementTy.isa<mlir::FloatType>()) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APFloat::getZero(
                         elementTy.cast<mlir::FloatType>().getFloatSemantics(),
                         /*negative=*/false)});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    } else if (elementTy.isa<mlir::IntegerType>() &&
               elementTy.getIntOrFloatBitWidth() != 8) {
      auto constAttr = DenseElementsAttr::get(
          constType, {APInt::getZero(elementTy.getIntOrFloatBitWidth())});
      return rewriter.create<stablehlo::ConstantOp>(op->getLoc(), constType,
                                                    constAttr);
    }
  }

  op->emitError("unimplemented lowering in AtenIndirectDataMovementOp");
  return nullptr;
}

//ConvertAtenEmbeddingBagPaddingIdxOp
template <>
LogicalResult ConvertAtenOp<AtenEmbeddingBagPaddingIdxOp>::matchAndRewrite(
	AtenEmbeddingBagPaddingIdxOp op, OpAdaptor adaptor, ConversionPatternRewriter &rewriter) const {

    std::cout << "0";

	Location loc = op->getLoc();
    auto context = op->getContext();
    Value weight = adaptor.getWeight();
    Value indices = adaptor.getIndices();
    Value offsets = adaptor.getOffsets();
    Value scaleGradByFreq = op.getScaleGradByFreq();
    Value mode = op.getMode();
    Value sparse = op.getSparse();
    Value includeLastOffset = op.getIncludeLastOffset();

    // Convert Inputs to their correct types

    std::cout << "1";

    bool scaleGradByFreqBool;
    if (!matchPattern(scaleGradByFreq,
                      m_TorchConstantBool(&scaleGradByFreqBool))) {
      return rewriter.notifyMatchFailure(
          op, "scale_grad_by_freq is expected to be a constant boolean value.");
    }

    if (scaleGradByFreqBool) {
      return rewriter.notifyMatchFailure(
          op, "Unimplemented: scale_grad_by_freq=True.");
    }

    std::cout << "2";

    int64_t modeInt;
    if (!matchPattern(mode, m_TorchConstantInt(&modeInt))) {
      return rewriter.notifyMatchFailure(
          op, "mode is expected to be a constant integer value.");
    }

    if (modeInt != torch_upstream::EmbeddingBagMode::MODE_SUM) {
      return rewriter.notifyMatchFailure(
          op,
          "Unimplemented: Mean and Max mode are not supported yet for EmbeddingBag.");
    }

    std::cout << "3";

    bool isSparse;
    if (!matchPattern(sparse, m_TorchConstantBool(&isSparse))) {
      return rewriter.notifyMatchFailure(
          op, "sparse is expected to be a constant boolean value.");
    }

    if (isSparse) {
      return rewriter.notifyMatchFailure(
          op,
          "Unimplemented: Sparse mode is not supported yet for EmbeddingBag.");
    }

    bool discardLastOffset;
    if (!matchPattern(includeLastOffset,
                      m_TorchConstantBool(&discardLastOffset))) {
      return rewriter.notifyMatchFailure(
          op,
          "include_last_offset is expected to be a constant boolean value.");
    }

    std::cout << "4";

    auto weightTy = weight.getType().cast<RankedTensorType>();

    if (weightTy.getRank() != 2)
      return rewriter.notifyMatchFailure(op, "weight must be rank 2");

    auto indicesTy = indices.getType().cast<RankedTensorType>();
    if (indicesTy.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "indices must be a vector");

    auto offsetsTy = offsets.getType().cast<RankedTensorType>();
    if (offsetsTy.getRank() != 1)
      return rewriter.notifyMatchFailure(op, "offsets much be a vector");

    Type weightElemTy = weightTy.getElementType();

    std::cout << "5";

    int64_t iterationMapDimension = weightTy.getRank() + indicesTy.getRank();
    SmallVector<AffineExpr> indicesExpr;
    indicesExpr.push_back(mlir::getAffineDimExpr(1, context));
    auto indicesIndexingMap =
          AffineMap::get(/*dimCount=*/iterationMapDimension, /*symbolCount=*/0,
                         indicesExpr, context);

    std::cout << "6";

    SmallVector<AffineExpr> offsetsExpr;
    offsetsExpr.push_back(mlir::getAffineDimExpr(0, context));

    auto offsetIndexingMap =
          AffineMap::get(/*dimCount=*/iterationMapDimension, /*symbolCount=*/0,
                         offsetsExpr, context);

    SmallVector<AffineExpr> outputExpr;
    outputExpr.push_back(mlir::getAffineDimExpr(0, context));
    outputExpr.push_back(mlir::getAffineDimExpr(2, context));

    auto outputIndexingMap =
          AffineMap::get(/*dimCount=*/iterationMapDimension, /*symbolCount=*/0,
                         outputExpr, context);

    SmallVector<AffineMap, 3> indexingMaps = {
          indicesIndexingMap,
          offsetIndexingMap,
          outputIndexingMap,
    };

    // Reduce along the indices dim
    SmallVector<utils::IteratorType> iteratorTypes(
        {utils::IteratorType::parallel, utils::IteratorType::reduction,
         utils::IteratorType::parallel});

    Value embeddingDim = getDimOp(rewriter, loc, weight, 1);
    Value initTensor;
    Value offsetsLength;
    Value indicesLength;
    Value weightsLength;
    if (!discardLastOffset) {
      SmallVector<Value> sizes{getDimOp(rewriter, loc, offsets, 0),
                                 embeddingDim};

      initTensor = createZeroInitTensor(rewriter, loc, sizes, weightElemTy);
      offsetsLength = getDimOp(rewriter, loc, offsets, 0);
      indicesLength = getDimOp(rewriter, loc, indices, 0);
      weightsLength = getDimOp(rewriter, loc, weight, 1);
    } else {
      return rewriter.notifyMatchFailure(
            op, "Unimplemented: include last offset is not yet "
                "supported for EmbeddingBag.");
    }

    // WORK STARTS HERE

    std::cout << "7";
    
    SmallVector<Value> outputBagShape;
    SmallVector<Value> offsetBagShape2;
    SmallVector<Value> bagSizeShape;
    SmallVector<Value> maxIndicesShape;

    outputBagShape.push_back(offsetsLength);
    outputBagShape.push_back(weightsLength);

    Value zeroDim = rewriter.create<arith::ConstantIndexOp>(loc, /*value=*/0);
    offsetBagShape2.push_back(zeroDim);

    //TODO It seems like we want to copy offsets, but the values aren't used for dlrm

    bagSizeShape.push_back(offsetsLength);
    maxIndicesShape.push_back(offsetsLength);

    auto outputBagShape_ = rewriter.create<mlir::tensor::FromElementsOp>(
      op->getLoc(), outputBagShape);
    auto offsetBagShape2_ = rewriter.create<mlir::tensor::FromElementsOp>(
      op->getLoc(), offsetBagShape2);
    auto bagSizeShape_ = rewriter.create<mlir::tensor::FromElementsOp>(
      op->getLoc(), bagSizeShape);
    auto maxIndicesShape_ = rewriter.create<mlir::tensor::FromElementsOp>(
      op->getLoc(), maxIndicesShape);
    

    //auto placeholderOp1 = rewriter.create<stablehlo::SqrtOp>(loc, weight);    
    //rewriter.replaceOp(op, placeholderOp1.getResult());

    std::cout << "8";

    auto restulType0 = typeConverter->convertType(op->getResult(0).getType());
    Value castedEmbeddingBagResult =
        rewriter.create<tensor::CastOp>(loc, restulType0, outputBagShape_);

    SmallVector<Value> offsetResultSize;
    Type offsetElemTy = offsetsTy.getElementType();
    offsetResultSize.push_back(zeroDim);
    Value offsetResult = rewriter.create<tensor::EmptyOp>(
        loc, getAsOpFoldResult(offsetResultSize), offsetElemTy);
    auto resultType1 = typeConverter->convertType(op->getResult(1).getType());
    Value castedOffsetResult =
        rewriter.create<tensor::CastOp>(loc, resultType1, offsetResult);

    SmallVector<Value> offsetSize = getTensorSizes(rewriter, loc, offsets);
    // bagsize, vector of size offset with zeros, I think this is always just
    // a vector of zeros in the sum mode
    Value bagSize =
        createZeroInitTensor(rewriter, loc, offsetSize, offsetElemTy);
    auto resultType2 = typeConverter->convertType(op->getResult(2).getType());
    Value castedBagSizeResult =
        rewriter.create<tensor::CastOp>(loc, resultType2, bagSize);

    // max indices, vector of size offset with zeros, this is also always a
    // vector of zeros in the sum mode. Its mainly used in the max mode.
    Value indicesOut =
        createZeroInitTensor(rewriter, loc, offsetSize, offsetElemTy);
    auto resultType3 = typeConverter->convertType(op->getResult(3).getType());
    Value castedMaxIndices =
        rewriter.create<tensor::CastOp>(loc, resultType3, indicesOut);
    

    rewriter.replaceOp(op, ValueRange{castedEmbeddingBagResult, castedOffsetResult,
                              castedBagSizeResult, castedMaxIndices});

    


    return success();
  }


void mlir::torch::torch_to_stablehlo::populateIndirectDataMovementPatternsAndLegality(
        TypeConverter &typeConverter, RewritePatternSet &patterns,
        ConversionTarget &target, const TorchToStablehloOptions &options) {
  MLIRContext *context = patterns.getContext();
  target.addIllegalOp<AtenEmbeddingBagPaddingIdxOp>();
  patterns.add<ConvertAtenOp<AtenEmbeddingBagPaddingIdxOp>>(typeConverter, context, options);
}
