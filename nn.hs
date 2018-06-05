{-# LANGUAGE FlexibleContexts #-}
{-# LANGUAGE ExistentialQuantification #-}
{-# LANGUAGE RankNTypes #-}
--{-# LANGUAGE GADTs #-}

import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel as V
import System.Random.MWC as Random
import Control.Monad (forM_)
import Numeric.AD.Mode.Reverse as AD
import Numeric.AD.Internal.Reverse (Reverse, Tape)
import Data.Reflection (Reifies)
    
-- Contem a matrix, o vetor de vieses e a função de ativação
data PreLayer a =  PreL (Vector a) (Matrix a) (a -> a)
-- Contem a, z e uma précamada
data Layer a = Lay (Vector a) (Vector a) (PreLayer a)

data ShowBox = forall a. Show a => SB a (forall s. Reifies s Tape =>
                                         Reverse s a -> Reverse s a)

-- sigmoid :: R -> R
sigmoid x = 1 / (1 + (exp (-x)))
            
-- relu :: R -> R
relu x
  | x > 0 = x
  | otherwise = 0
                
mapVector f = mapVectorWithIndex (\ i x -> f x)


changeMatrix mat = runSTMatrix $ do
  m <- thawMatrix mat
  forM_ [(i, j) | i <- [0..10], j <- [0..10]] $ \(i,j) -> writeMatrix m i j 0
  return m

sumIntoMatrix mat1 mat2 = runSTMatrix $ do
  m1 <- thawMatrix mat1
  m2 <- thawMatrix mat2
  forM_ [(i, j) | i <- [0..1], j <- [0..1]] $ \(i,j) -> do
                                          aij <- readMatrix m2 i j
                                          modifyMatrix m1 i j (+ aij)
  return m1

-- forward :: Vector R -> [PreLayer R] -> ([Layer R], Vector R)
forward v [] = ([], v)
forward v ((PreL b w act):xs) = let z = (flatten (w <> (asColumn v) + (asColumn b))) in
                                -- let v' = mapVector act z in
                                let (r, o) = forward v xs in
                                ((Lay z v (PreL b w act)):r, o)

-- random generators
randomMat m n gen = (mapMatrixWithIndexM (\ _ _ -> uniformR (-1, 1) gen) (konst (1.0 :: Double) (m, n))) :: IO (Matrix R)
randomVec n gen = (mapVectorWithIndexM (\ _ _ -> uniformR (-1, 1) gen) (konst 1 n :: Vector Double)) :: IO (Vector R)

-- create [PreLayer R] from list of layers
-- input: [input_dim, hidden_layers_dim, output_dim]
-- setupNetwork :: [Int] -> [PreLayer R]
setupNetwork [a] _ _ = return []
setupNetwork (x:xs) (act:acts) gen = do
  let y = head xs
  b <- (randomVec y gen)
  m <- (randomMat y x gen)
  list <- (setupNetwork xs acts gen)
  return $ (PreL b m act):list

--backPropagation :: 
-- train :: [a * b] -> [Layer R] -> [Layer R]
-- train [] = id
-- train ((input, output):xs) nn = 

calculateLayerGrad act' mat z a' = let newBias = (zipVectorWith (\ x y -> x + (act' y)) a' z) in
                                   ((tr mat) <> (asColumn newBias), newBias)

-- -- backPropagation :: [Layer R] -> [Layer R]
backPropagation [] = []
backPropagation ((Lay z a (PreL bias mat act)):xs) = (Lay z (flatten a') (PreL bias' mat' act)) : xs'
    where
      xs' = backPropagation xs
      (Lay z a'' _) = head xs
      (a', bias') = calculateLayerGrad (diff sigmoid) mat z a''
      mat' = (asColumn a) <> (asRow bias')

-- main :: IO ()
-- main = do
--   gen <- create
--   input <- randomVec 10 gen
--   layers <- setupNetwork [10, 100, 2] [relu, sigmoid] gen
--   print $ snd(forward input layers)
