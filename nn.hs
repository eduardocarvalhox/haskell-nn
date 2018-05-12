import Numeric.LinearAlgebra
import Numeric.LinearAlgebra.Devel as V
import System.Random.MWC as Random
import Control.Monad (forM_)


data PreLayer a = PreL (Vector a) (Matrix a) (a -> a)
data Layer a = Lay (Vector a) (PreLayer a)

sigmoid :: R -> R
sigmoid x = 1 / (1 + (exp (-x)))

relu :: R -> R
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
  forM_ [(i, j) | i <- [0..1], j <- [0..1]] $ \(i,j) -> modifyMatrix m1 i j (+ readMatrix m2 i j)
  return m1

forward :: Vector R -> [PreLayer R] -> ([Layer R], Vector R)
forward v [] = ([], v)
forward v ((PreL b w act):xs) = let v' = mapVector act (flatten (w <> (asColumn v) + (asColumn b))) in
                                let (r, o) = forward v' xs in
                                ((Lay v (PreL b w act)):r, o)

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

main :: IO ()
main = do
  gen <- create
  input <- randomVec 10 gen
  layers <- setupNetwork [10, 100, 2] [relu, sigmoid] gen
  print $ snd(forward input layers)
