/*!
 * Start Bootstrap - Agency Bootstrap Theme (http://startbootstrap.com)
 * Code licensed under the Apache License v2.0.
 * For details, see http://www.apache.org/licenses/LICENSE-2.0.
 */

// jQuery for page scrolling feature - requires jQuery Easing plugin
$(function() {
    $('a.page-scroll').bind('click', function(event) {
        var $anchor = $(this);
        $('html, body').stop().animate({
            scrollTop: $($anchor.attr('href')).offset().top
        }, 1500, 'easeInOutExpo');
        event.preventDefault();
    });
});

// Highlight the top nav as scrolling occurs
$('body').scrollspy({
    target: '.navbar-fixed-top'
})

// Closes the Responsive Menu on Menu Item Click
$('.navbar-collapse ul li a').click(function() {
    $('.navbar-toggle:visible').click();
});



///////////////

// external js: masonry.pkgd.js, imagesloaded.pkgd.js

docReady( function() {
  // init Isotope
  var grid = document.querySelector('.grid');

  var msnry = new Masonry( grid, {
    itemSelector: '.grid-item',
    columnWidth: '.grid-sizer',
    percentPosition: true
  });

  // imagesLoaded( grid, function() {
  //   // layout Masonry after each image loads
  //   msnry.layout();
  // });



});

// init Masonry
var $grid = $('.grid').masonry({
  // options...
});
// layout Masonry after each image loads
$grid.imagesLoaded().progress( function() {
  $grid.masonry('layout');
});
