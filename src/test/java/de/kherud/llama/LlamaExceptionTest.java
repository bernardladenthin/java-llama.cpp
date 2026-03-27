package de.kherud.llama;

import org.junit.Test;

import static org.junit.Assert.*;

public class LlamaExceptionTest {

	@Test
	public void testMessageIsPreserved() {
		LlamaException ex = new LlamaException("something went wrong");
		assertEquals("something went wrong", ex.getMessage());
	}

	@Test
	public void testIsRuntimeException() {
		LlamaException ex = new LlamaException("error");
		assertTrue(ex instanceof RuntimeException);
	}

	@Test
	public void testEmptyMessage() {
		LlamaException ex = new LlamaException("");
		assertEquals("", ex.getMessage());
	}

	@Test
	public void testNullMessage() {
		LlamaException ex = new LlamaException(null);
		assertNull(ex.getMessage());
	}

	@Test
	public void testCanBeThrown() {
		boolean caught = false;
		try {
			throw new LlamaException("thrown");
		} catch (LlamaException e) {
			assertEquals("thrown", e.getMessage());
			caught = true;
		}
		assertTrue("Expected LlamaException to be thrown", caught);
	}
}
