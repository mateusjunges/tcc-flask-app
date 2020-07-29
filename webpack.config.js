const webpack = require('webpack');
const resolve = require('path').resolve;


module.exports {
 entry: __dirname + '/assets/js/index.js',
 output:{
      path: resolve('public'),
      filename: 'bundle.js',
      publicPath: resolve('public')
 },
 plugins: [
 ],
 node: {
    fs: 'empty',
 },
 resolve: {
  extensions: ['.js', '.css', '.html']
 }
};